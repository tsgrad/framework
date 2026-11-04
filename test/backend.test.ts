import { expect, test } from 'vitest';
import { getBackendKind, initBackend, setBackendForTesting } from '../src/backend/index.js';
import { tryCreateWebGpuBackend } from '../src/backend/webgpu/webgpu_backend.js';

const webgpuAvailable = await tryCreateWebGpuBackend();

test('explicit CPU backend selection is deterministic', async () => {
    setBackendForTesting(null);
    await initBackend({ backend: 'cpu' });
    expect(getBackendKind()).toBe('cpu');
});

test.skipIf(!webgpuAvailable)('explicit WebGPU selection uses the Node provider when available', async () => {
    setBackendForTesting(webgpuAvailable);
    const backend = await initBackend({ backend: 'webgpu' });
    expect(backend.kind).toBe('webgpu');
    await backend.synchronize();
});
