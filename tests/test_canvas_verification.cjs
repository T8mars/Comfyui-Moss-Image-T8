const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const os = require('node:os');
const path = require('node:path');
const { validateHistory, waitForHistory, assertPng } = require('../scripts/verify_canvas_run.cjs');

function history() {
  return { status: { status_str: 'success', completed: true, messages: [['execution_cached', { nodes: ['1'] }]] },
           outputs: { '7': { images: [{ filename: 'image.png' }] } } };
}

test('only completed, single-image, fresh sampler history passes', () => {
  assert.equal(validateHistory(history()).length, 1);
  const incomplete = history(); incomplete.status.completed = false;
  assert.throws(() => validateHistory(incomplete), /completed/);
  const empty = history(); empty.outputs['7'].images = [];
  assert.throws(() => validateHistory(empty), /exactly one/);
  const absent = history(); absent.status.messages = [];
  assert.throws(() => validateHistory(absent), /fresh sampler/);
  const cached = history(); cached.status.messages.push(['execution_cached', { nodes: [5] }]);
  assert.throws(() => validateHistory(cached), /fresh sampler/);
});

test('a displayable JPEG cannot be saved as PNG evidence', async () => {
  const png = await fs.readFile(path.join(__dirname, '../example_workflows/moss-image-sft.png'));
  assert.doesNotThrow(() => assertPng(png));
  assert.throws(() => assertPng(Buffer.from('ffd8ffe000104a464946000101', 'hex')), /PNG/);
  assert.throws(() => assertPng(png.subarray(0, 7)), /PNG/);
});

test('missing history expires without another prompt submission', async () => {
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'moss-verify-'));
  const calls = [];
  const request = { get: async (url, options) => {
    calls.push({ url, options });
    return { ok: () => true, json: async () => url.endsWith('/queue')
      ? { queue_running: [[1, 'same-id', { prompt: 'not included' }]], queue_pending: [] } : {} };
  } };
  try {
    await assert.rejects(waitForHistory(request, 'same-id', { folder, timeout: 0.03, pollMs: 10 }), /same-id/);
    const report = JSON.parse(await fs.readFile(path.join(folder, 'timeout.json')));
    assert.deepEqual(report.queue_running, ['same-id']);
    assert.ok(calls.filter(c => c.url.includes('/history/')).every(c => c.options.timeout <= 30));
    assert.ok(calls.every(c => !c.url.endsWith('/prompt')));
  } finally { await fs.rm(folder, { recursive: true, force: true }); }
});

test('transport timeout writes prompt and queue diagnostics', async () => {
  const folder = await fs.mkdtemp(path.join(os.tmpdir(), 'moss-verify-'));
  const request = { get: async url => {
    if (!url.endsWith('/queue')) throw new Error('apiRequestContext.get: Timeout 20ms exceeded.');
    return { ok: () => true, json: async () => ({ queue_pending: [[1, 'pending-id']] }) };
  } };
  try {
    await assert.rejects(waitForHistory(request, 'same-id', { folder, timeout: 0.02 }), /same-id/);
    const report = JSON.parse(await fs.readFile(path.join(folder, 'timeout.json')));
    assert.deepEqual(report.queue_pending, ['pending-id']);
  } finally { await fs.rm(folder, { recursive: true, force: true }); }
});
