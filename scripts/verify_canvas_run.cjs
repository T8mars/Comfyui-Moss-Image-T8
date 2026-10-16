// End-to-end acceptance: load the delivered GUI JSON, CLICK Run, inspect the
// browser's actual POST /prompt, await server success and a rendered preview.
const fs = require('node:fs/promises');
const path = require('node:path');
const crypto = require('node:crypto');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

const root = path.resolve(__dirname, '..');
const baseURL = process.env.MOSS_COMFY_URL || 'http://127.0.0.1:8279';
const variants = process.argv.slice(2).length ? process.argv.slice(2) : ['sft', 'base'];
const digest = data => crypto.createHash('sha256').update(data).digest('hex');
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function checkPrompt(actual, expected) {
  assert.deepEqual(Object.keys(actual).sort(), Object.keys(expected).sort(), 'Unexpected node set');
  for (const [id, node] of Object.entries(expected)) {
    assert.equal(actual[id].class_type, node.class_type, `node ${id} class`);
    assert.deepEqual(actual[id].inputs, node.inputs, `node ${id} inputs`);
  }
}

async function main() {
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
      let history;
      let lastProgress = 0;
      for (;;) {
        const status = await page.request.get(`${baseURL}/history/${promptId}`);
        assert.ok(status.ok());
        const data = await status.json();
        if (data[promptId]) { history = data[promptId]; break; }
        if (Date.now() - lastProgress > 30000) {
          const progress = await page.evaluate(() => {
            const app = window.comfyAPI.app.app;
            return { runningNodeId: app.runningNodeId, progress: app.progress || null };
          });
          console.log(`LIVE ${variant}: ${JSON.stringify(progress)}`);
          await page.screenshot({ path: path.join(folder, 'canvas-running.png') });
          lastProgress = Date.now();
        }
        // This remains tied to this live prompt ID. An observation delay never
        // causes the model to be restarted or the prompt to be submitted again.
        await sleep(2000);
      }
      await fs.writeFile(path.join(folder, 'history.json'), JSON.stringify(history, null, 2));
      assert.equal(history.status.status_str, 'success', JSON.stringify(history.status));
      checkPrompt(history.prompt[2], expected);
      const cached = history.status.messages.find(([kind]) => kind === 'execution_cached')?.[1]?.nodes || [];
      assert.ok(!cached.includes('5'), 'Sampler was cached; this does not prove a fresh model run');
      const images = history.outputs['7'].images;
      assert.equal(images.length, 1);
      for (const image of images) {
        const result = await page.request.get(`${baseURL}/view?${new URLSearchParams(image)}`);
        assert.ok(result.ok());
        await fs.writeFile(path.join(folder, image.filename), await result.body());
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

main().catch(error => { console.error(error); process.exitCode = 1; });
