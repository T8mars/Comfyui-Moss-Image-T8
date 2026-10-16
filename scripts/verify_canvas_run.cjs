// End-to-end acceptance: load the delivered GUI JSON, CLICK Run, inspect the
// browser's actual POST /prompt, await server success and a rendered preview.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');

const root = path.resolve(__dirname, '..');
const baseURL = (process.env.MOSS_COMFY_URL || 'http://127.0.0.1:8279').replace(/\/+$/, '');
const variants = process.argv.slice(2).length ? process.argv.slice(2) : ['sft', 'base'];
const digest = data => crypto.createHash('sha256').update(data).digest('hex');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const timeoutSeconds = Number(process.env.MOSS_VERIFY_TIMEOUT_SECONDS || 1800);
assert.ok(Number.isFinite(timeoutSeconds) && timeoutSeconds > 0, 'MOSS_VERIFY_TIMEOUT_SECONDS must be finite and positive');

function checkPrompt(actual, expected) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), 'Unexpected node set');
  for (const [id, node] of Object.entries(expected)) {
    assert.equal(actual[id].class_type, node.class_type, `node ${id} class`);
    assert.deepEqual(actual[id].inputs, node.inputs, `node ${id} inputs`);
  }
}

function validateHistory(history) {
  assert.equal(history.status?.status_str, 'success', JSON.stringify(history.status));
  assert.equal(history.status.completed, true, 'Server history is not completed');
  const events = (history.status.messages || []).filter(([kind]) => kind === 'execution_cached');
  assert.ok(events.length && events.every(([, data]) => !(data.nodes || []).map(String).includes('5')),
            'A fresh sampler execution was not established');
  const images = history.outputs?.['7']?.images;
  assert.ok(Array.isArray(images) && images.length === 1, 'Expected exactly one SaveImage output');
  return images;
}

function assertPng(bytes) {
  assert.ok(bytes.subarray(0, 8).equals(Buffer.from('89504e470d0a1a0a', 'hex')),
            'Expected a PNG image from SaveImage');
}

async function waitForHistory(requestContext, promptId, { folder, start = Date.now(),
                         timeout = timeoutSeconds, onProgress = async () => {}, pollMs = 2000 } = {}) {
  const deadline = start + timeout * 1000;
  async function failTimeout(cause) {
    const report = { prompt_id: promptId, timeout_seconds: timeout };
    try {
      const response = await requestContext.get(`${baseURL}/queue`, { timeout: 5000 });
      assert.ok(response.ok());
      const queue = await response.json();
      for (const name of ['queue_running', 'queue_pending']) report[name] = (queue[name] || []).map(item => item[1]);
    } catch (error) { report.queue_error = String(error); }
    await fs.writeFile(path.join(folder, 'timeout.json'), JSON.stringify(report, null, 2));
    throw new Error(`Timed out waiting for prompt ${promptId}; inspect its server queue/history before rerunning`, { cause });
  }
  for (;;) {
    const remaining = deadline - Date.now();
    if (remaining <= 0) await failTimeout();
    let status;
    try {
      status = await requestContext.get(`${baseURL}/history/${promptId}`, { timeout: Math.min(30000, remaining) });
    } catch (error) {
      if (error.name === 'TimeoutError' || /Timeout .* exceeded/.test(String(error))) await failTimeout(error);
      throw error;
    }
    assert.ok(status.ok());
    const data = await status.json();
    if (data[promptId]) return data[promptId];
    await onProgress();
    // Keep observing this prompt; never cancel or resubmit on an observation timeout.
    await sleep(Math.max(0, Math.min(pollMs, deadline - Date.now())));
  }
}

async function main() {
  const { chromium } = require('playwright');
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    const pageErrors = [];
    page.on('pageerror', error => pageErrors.push(String(error)));
    await page.goto(baseURL, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.comfyAPI?.app?.app?.graph?._nodes?.length > 0,
                              undefined, { timeout: 60000 });
    for (const variant of variants) {
      assert.ok(['base', 'sft'].includes(variant));
      const folder = path.join(root, 'artifacts', `canvas-${variant}`);
      await fs.mkdir(folder, { recursive: true });
      const raw = await fs.readFile(path.join(root, `example_workflows/moss-image-${variant}.json`));
      const workflow = JSON.parse(raw.toString('utf8'));
      const expected = JSON.parse(await fs.readFile(path.join(root, `example_workflows/moss-image-${variant}-api.json`), 'utf8'));
      const before = await page.evaluate(async workflow => {
        const { app } = await import('/scripts/app.js');
        await app.loadGraphData(workflow);
        const converted = await app.graphToPrompt();
        return { prompt: converted.output,
                 nodes: app.graph._nodes.map(n => ({ id: n.id, type: n.type, has_errors: !!n.has_errors })),
                 sampler_widgets: app.graph.getNodeById(5).widgets.map(w => ({ name: w.name, value: w.value })) };
      }, workflow);
      checkPrompt(before.prompt, expected);
      assert.equal(before.nodes.length, 7);
      assert.ok(before.nodes.every(n => !n.has_errors), 'Canvas contains validation errors');
      assert.equal(before.sampler_widgets.find(w => w.name === 'control_after_generate').value, 'fixed');
      await fs.writeFile(path.join(folder, 'canvas-before.json'), JSON.stringify(before, null, 2));
      await page.screenshot({ path: path.join(folder, 'canvas-before.png') });

      const runButton = page.getByRole('button', { name: /^(运行|Run)(\s|$)/ }).first();
      await runButton.waitFor({ state: 'visible', timeout: 15000 });
      const buttonText = await runButton.innerText();
      const responsePromise = page.waitForResponse(response =>
        /^\/(?:api\/)?prompt$/.test(new URL(response.url()).pathname)
          && response.request().method() === 'POST', { timeout: 60000 });
      const start = Date.now();
      // This is the user-visible canvas button, not an API POST or app.queuePrompt.
      await runButton.click();
      const response = await responsePromise;
      const request = response.request().postDataJSON();
      const submission = await response.json();
      await fs.writeFile(path.join(folder, 'browser-request.json'), JSON.stringify(request, null, 2));
      await fs.writeFile(path.join(folder, 'submission.json'), JSON.stringify(submission, null, 2));
      assert.ok(response.ok(), JSON.stringify(submission));
      assert.ok(submission.prompt_id && !Object.keys(submission.node_errors || {}).length);
      checkPrompt(request.prompt, expected);
      const embedded = request.extra_data?.extra_pnginfo?.workflow;
      assert.ok(embedded?.nodes?.length === 7, 'Canvas did not include its workflow for PNG metadata');
      const promptId = submission.prompt_id;
      console.log(`CLICKED Run: ${variant}, ${promptId}, 1024x1024 / 50 / CFG3 / shift5 / skip22`);
      let lastProgress = 0;
      const history = await waitForHistory(page.request, promptId, { folder, start, onProgress: async () => {
        if (Date.now() - lastProgress > 30000) {
          const progress = await page.evaluate(() => {
            const app = window.comfyAPI.app.app;
            return { runningNodeId: app.runningNodeId, progress: app.progress || null };
          });
          console.log(`LIVE ${variant}: ${JSON.stringify(progress)}`);
          await page.screenshot({ path: path.join(folder, 'canvas-running.png') });
          lastProgress = Date.now();
        }
      } });
      await fs.writeFile(path.join(folder, 'history.json'), JSON.stringify(history, null, 2));
      checkPrompt(history.prompt[2], expected);
      const images = validateHistory(history);
      for (const image of images) {
        const result = await page.request.get(`${baseURL}/view?${new URLSearchParams(image)}`);
        assert.ok(result.ok());
        const bytes = await result.body();
        assertPng(bytes);
        await fs.writeFile(path.join(folder, image.filename), bytes);
      }
      await page.waitForFunction(() => {
        const node = window.comfyAPI.app.app.graph.getNodeById(7);
        return node?.imgs?.some(image => image.complete && image.naturalWidth === 1024 && image.naturalHeight === 1024);
      }, undefined, { timeout: 30000 });
      const after = await page.evaluate(() => {
        const { app } = window.comfyAPI.app;
        const node = app.graph.getNodeById(7);
        return { sampler_widgets: app.graph.getNodeById(5).widgets.map(w => ({ name: w.name, value: w.value })),
                 preview_sizes: node.imgs.map(image => [image.naturalWidth, image.naturalHeight]),
                 node_outputs: app.nodeOutputs?.['7'] || null };
      });
      assert.equal(after.sampler_widgets.find(w => w.name === 'seed').value, 42);
      const record = { variant, source: 'user-visible canvas Run button click', button_text: buttonText,
                       workflow_sha256: digest(raw), prompt_id: promptId, elapsed_seconds: (Date.now() - start) / 1000,
                       sampler_executed_fresh: true, preview_verified: true, images, before, after, page_errors: pageErrors };
      await fs.writeFile(path.join(folder, 'canvas-proof.json'), JSON.stringify(record, null, 2));
      await page.screenshot({ path: path.join(folder, 'canvas-success.png') });
      console.log(`CANVAS PASS ${variant}: success, fresh sampler, 1024px preview, fixed seed42, workflow attached to PNG`);
    }
  } finally {
    await browser.close();
  }
}

module.exports = { validateHistory, waitForHistory, assertPng };
if (require.main === module) main().catch(error => { console.error(error); process.exitCode = 1; });
