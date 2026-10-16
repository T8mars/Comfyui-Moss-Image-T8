// Requires Playwright in NODE_PATH, and a locally running isolated ComfyUI.
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(__dirname, '..');
  const browser = await chromium.launch({ channel: 'msedge', headless: true, args: ['--disable-gpu'] });
  try {
    const page = await browser.newPage({ viewport: { width: 1920, height: 1080 } });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(process.env.MOSS_COMFY_URL || 'http://127.0.0.1:8279', { waitUntil: 'domcontentloaded' });
    // Wait until startup has restored its default graph; loading before that
    // races the frontend's asynchronous initial workflow restoration.
    await page.waitForFunction(() => window.comfyAPI?.app?.app?.graph?._nodes?.length > 0,
                              undefined, { timeout: 60000 });
    for (const variant of ['base', 'sft']) {
      const workflow = JSON.parse(await fs.readFile(path.join(root, `example_workflows/moss-image-${variant}.json`), 'utf8'));
      const result = await page.evaluate(async workflow => {
        const { app } = await import('/scripts/app.js');
        await app.loadGraphData(workflow);
        const converted = await app.graphToPrompt();
        return { nodes: app.graph._nodes.map(node => ({ id: node.id, type: node.type, has_errors: !!node.has_errors })),
                 prompt: converted.output };
      }, workflow);
      const expected = JSON.parse(await fs.readFile(path.join(root, `example_workflows/moss-image-${variant}-api.json`), 'utf8'));
      await fs.writeFile(path.join(root, `artifacts/ui-${variant}.json`), JSON.stringify({ ...result, browser_errors: errors }, null, 2));
      await page.screenshot({ path: path.join(root, `artifacts/ui-${variant}.png`) });
      for (const [id, node] of Object.entries(expected)) {
        if (JSON.stringify(result.prompt[id]?.inputs) !== JSON.stringify(node.inputs)) {
          // Compare key order independently while keeping scalar/array values exact.
          for (const [key, value] of Object.entries(node.inputs)) {
            if (JSON.stringify(result.prompt[id]?.inputs[key]) !== JSON.stringify(value)) {
              throw new Error(`GUI/API mismatch: ${variant} node ${id}.${key}: ${JSON.stringify(result.prompt[id]?.inputs[key])}`);
            }
          }
        }
      }
      if (result.nodes.length !== 7 || result.nodes.some(node => node.has_errors)) {
        throw new Error('GUI node validation errors; check node registration and selected model files.');
      }
      await fs.writeFile(path.join(root, `artifacts/ui-${variant}.json`), JSON.stringify({ ...result, browser_errors: errors }, null, 2));
      await page.screenshot({ path: path.join(root, `artifacts/ui-${variant}.png`) });
      console.log(`GUI verified: ${variant}, 7 nodes; exported API matches all inputs.`);
    }
  } finally {
    await browser.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
