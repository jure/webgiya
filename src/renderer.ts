import * as THREE from 'three/webgpu';
import { Inspector } from 'three/addons/inspector/Inspector.js';

export type RendererBundle = {
  renderer: THREE.WebGPURenderer;
  container: HTMLDivElement;
};

export async function initRenderer(containerSelector = '#app'):
  Promise<RendererBundle> {
  const container = document.querySelector<HTMLDivElement>(containerSelector);
  if (!container) throw new Error('App container not found');
  const renderer = new THREE.WebGPURenderer({ forceWebGL: false, antialias: true, requiredLimits: {
    // maxColorAttachmentBytesPerSample: 48,
    maxStorageBuffersPerShaderStage: 10,
    maxComputeWorkgroupSizeX: 1024,
    maxComputeInvocationsPerWorkgroup: 1024,
    // maxStorageBufferBindingSize: 4294967292
  }});

  renderer.inspector = new Inspector();
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.toneMappingExposure = 1
  container.appendChild(renderer.domElement);
  await renderer.init();
  return { renderer, container };
}

