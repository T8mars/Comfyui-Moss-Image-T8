// Exercise the frontend's real file input parser for both JSON and PNG files.
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const { chromium } = require('playwright');
const root = path.resolve(__dirname, '..');

(async () => {
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    await page.goto(process.env.MOSS_COMFY_URL || 'http://127.0.0.1:8279', { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => window.comfyAPI?.app?.app?.graph?._nodes?.length > 0,
                              undefined, { timeout: 60000 });
    const inputs = await page.locator('input[type=file]').evaluateAll(elements => elements.map(e => ({ id: e.id, accept: e.accept })));
    assert.ok(inputs.length, 'No workflow file input found; inspect frontend before changing this verifier');
    console.log(`Observed file inputs: ${JSON.stringify(inputs)}`);
    const eligible = inputs.filter(item => item.id === 'comfy-file-input' && /json/i.test(item.accept) && /png/i.test(item.accept));
    assert.equal(eligible.length, 1, `Expected the observed Comfy JSON/PNG workflow input; observed ${JSON.stringify(inputs)}`);
    const input = page.locator('#comfy-file-input');
    const results = [];
    for (const method of ['file-input', 'canvas-drop']) {
    for (const variant of ['sft', 'base']) {
      const expected = JSON.parse(await fs.readFile(path.join(root, `example_workflows/moss-image-${variant}-api.json`), 'utf8'));
      for (const extension of ['json', 'png']) {
        const file = path.join(root, `example_workflows/moss-image-${variant}.${extension}`);
        // Clear the graph so an unchanged previous import cannot produce a false
        // positive while the asynchronous PNG parser is still running.
        await page.evaluate(() => window.comfyAPI.app.app.graph.clear());
        if (method === 'file-input') {
          await input.setInputFiles(file);
        } else {
          const buffer = await fs.readFile(file);
          const transfer = await page.evaluateHandle(({ name, mime, base64 }) => {
            const bytes = Uint8Array.from(atob(base64), value => value.charCodeAt(0));
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(new File([bytes], name, { type: mime }));
            return dataTransfer;
          }, { name: path.basename(file), mime: extension === 'png' ? 'image/png' : 'application/json', base64: buffer.toString('base64') });
          const canvas = await page.evaluateHandle(() => window.comfyAPI.app.app.canvas.canvas);
          await canvas.asElement().dispatchEvent('dragenter', { dataTransfer: transfer });
          await canvas.asElement().dispatchEvent('dragover', { dataTransfer: transfer });
          await canvas.asElement().dispatchEvent('drop', { dataTransfer: transfer });
          await transfer.dispose();
          await canvas.dispose();
        }
        await page.waitForFunction(({ variant }) => {
          const graph = window.comfyAPI.app.app.graph;
          const loader = graph.getNodeById(1);
          return graph._nodes.length === 7 && loader?.type === 'MOSSOfficialDiffusionLoader'
            && loader.widgets.find(w => w.name === 'model_name').value === `moss-image-${variant}-bf16.safetensors`;
        }, { variant }, { timeout: 30000 });
        const found = await page.evaluate(async () => {
          const { app } = await import('/scripts/app.js');
          const converted = await app.graphToPrompt();
          return { prompt: converted.output, nodes: app.graph._nodes.map(n => ({ id: n.id, type: n.type, has_errors: !!n.has_errors })) };
        });
        for (const [key, value] of Object.entries(expected)) {
          assert.equal(found.prompt[key].class_type, value.class_type);
          assert.deepEqual(found.prompt[key].inputs, value.inputs);
        }
        assert.ok(found.nodes.every(n => !n.has_errors));
        results.push({ method, variant, extension, input: inputs, all_node_inputs_exact: true, nodes: found.nodes });
        await page.screenshot({ path: path.join(root, `artifacts/import-${method}-${variant}-${extension}.png`) });
        await input.setInputFiles([]);
        console.log(`IMPORT PASS: ${method}, ${variant}.${extension}, 7 nodes, all parameters exact`);
      }
    }
    }
    await fs.writeFile(path.join(root, 'artifacts/canvas-import.json'), JSON.stringify(results, null, 2));
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
