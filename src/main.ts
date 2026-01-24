// main.ts
import './style.css';
import { initRenderer } from './renderer.ts';
import { createScene } from './scene.ts';
import { configureSceneSelector, createSceneSwitcher, createUI } from './ui.ts';
import {
  SCENE_PRESETS,
  captureDirectionalLightDefaults,
  clearSceneContent,
  recreateDirectionalLight,
  type SceneDefinition,
  type SceneSettings,
} from './content.ts';
import { createGBuffer } from './gbuffer.ts';
import { createSurfelPool } from './surfelPool.ts';
import { createSurfelPreparePass } from './surfelPreparePass.ts';
import { createSurfelAgePass } from './surfelAgePass.ts';
import { createSurfelFindMissingPass } from './surfelFindMissingPass.ts';
import { createSurfelAllocatePass } from './surfelAllocatePass.ts';
import { createSurfelDispatchArgs } from './surfelDispatchArgs.ts';
import { createSurfelHashGrid } from './surfelHashGrid.ts';
import {
  createSurfelScreenDebug,
  SCREEN_DEBUG_MODES,
} from './debug/surfelScreenDebug.ts';
import { MAX_SURFELS } from './constants.ts';

import { createSceneBVH, type SceneBVHBundle } from './sceneBvh.ts';
import { createSurfelIntegratePass } from './surfelIntegratePass.ts';
import { createSurfelGIResolvePass } from './surfelGIResolvePass.ts';
import * as THREE from 'three/webgpu';
import { mrt, output, pass, screenUV, texture, velocity } from 'three/tsl';
import {
  createLightControls,
  findSunPositionWeighted,
  setLightAnglesFromEnvMapSunUVLocation,
} from './lighting.ts';
import { fxaa } from 'three/examples/jsm/tsl/display/FXAANode.js';
import { traa } from 'three/examples/jsm/tsl/display/TRAANode.js';
import { createIntegratorDispatchArgs } from './integratorDispatchArgs.ts';
import {
  applyOcclusionSettings,
  configureRadialDepthGUI,
} from './surfelRadialDepth.ts';
import { EXRLoader, HDRLoader } from 'three/examples/jsm/Addons.js';

const loadingOverlay =
  document.querySelector<HTMLDivElement>('#loading-overlay');
const loadingMessage =
  document.querySelector<HTMLDivElement>('#loading-message');
const errorOverlay = document.querySelector<HTMLDivElement>('#error-overlay');
const errorMessage = document.querySelector<HTMLDivElement>('#error-message');

const WEBGPU_ERROR_MESSAGE =
  "WebGPU isn't supported in this browser. If you know WebGPU should work, it's also possible your browser needs a restart, as it's grown tired of all this computing.";

let hasFatalError = false;

function setOverlayVisible(element: HTMLDivElement | null, visible: boolean) {
  if (!element) return;
  element.classList.toggle('hidden', !visible);
}

function setLoading(message?: string) {
  if (loadingMessage && message) {
    loadingMessage.textContent = message;
  }
  setOverlayVisible(loadingOverlay, true);
}

function clearLoading() {
  setOverlayVisible(loadingOverlay, false);
}

function isWebGpuError(message: string) {
  const text = message.toLowerCase();
  return (
    text.includes('webgpu') ||
    text.includes('webgpurenderer') ||
    text.includes('webglbackend') ||
    text.includes('webglextensions') ||
    text.includes('getsupportedextensions') ||
    text.includes('context provider')
  );
}

function describeError(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string') return error;
  try {
    return JSON.stringify(error);
  } catch {
    return String(error);
  }
}

function showError(error: unknown) {
  if (hasFatalError) return;
  hasFatalError = true;
  const message = describeError(error);
  const friendly = isWebGpuError(message)
    ? WEBGPU_ERROR_MESSAGE
    : message || 'Unknown error';
  if (errorMessage) {
    errorMessage.textContent = friendly;
  }
  setOverlayVisible(errorOverlay, true);
  clearLoading();
}

setLoading('Initializing renderer');

window.addEventListener('error', (event) => {
  showError(event.error ?? event.message ?? event);
});

window.addEventListener('unhandledrejection', (event) => {
  showError(event.reason ?? event);
});

const { renderer } = await initRenderer().catch((error) => {
  showError(error);
  throw error;
});
const gui = createUI(renderer);
const sceneBundle = createScene(renderer);
const { scene, camera, controls, gbufferCamera } = sceneBundle;
let dirLight = sceneBundle.dirLight;
const dirLightDefaults = captureDirectionalLightDefaults(dirLight);
const defaultCameraSettings: NonNullable<SceneSettings['camera']> = {
  position: camera.position.clone(),
  target: controls.target.clone(),
};

// Load blue noise texture
const loader = new THREE.TextureLoader();
const blueNoise = await loader.loadAsync(
  `${import.meta.env.BASE_URL}textures/LDR_RGBA_0.png`,
);
blueNoise.colorSpace = THREE.NoColorSpace; // The default, but still...
blueNoise.wrapS = blueNoise.wrapT = THREE.RepeatWrapping;
blueNoise.minFilter = THREE.NearestFilter;
blueNoise.magFilter = THREE.NearestFilter;
blueNoise.generateMipmaps = false;

// EXR
// const pmremGenerator = new THREE.PMREMGenerator( renderer );
// pmremGenerator.compileEquirectangularShader();

const exrLoader = new EXRLoader();
exrLoader.setDataType(THREE.FloatType);
const hdrLoader = new HDRLoader();

const envCache = new Map<string, THREE.DataTexture>();

async function loadEnvTexture(path: string): Promise<THREE.DataTexture> {
  const cached = envCache.get(path);
  if (cached) return cached;

  const ext = path.split('.').pop()?.toLowerCase();
  const loader = ext === 'hdr' ? hdrLoader : exrLoader;
  const tex = (await loader.loadAsync(path)) as THREE.DataTexture;
  tex.generateMipmaps = true;
  tex.mapping = THREE.EquirectangularReflectionMapping;
  envCache.set(path, tex);
  return tex;
}

let envTex: THREE.DataTexture | null = null;

// Viewport camera can show debug
camera.layers.enable(1);
gbufferCamera.layers.set(0); // Everything but the debug stuff

const gbuffer = createGBuffer(renderer);

const surfelPool = createSurfelPool();
surfelPool.ensureCapacity(MAX_SURFELS);
const surfelPrepare = createSurfelPreparePass();
const surfelAge = createSurfelAgePass();
const surfelFindMissing = createSurfelFindMissingPass();
const surfelAllocate = createSurfelAllocatePass();
const surfelDispatchArgs = createSurfelDispatchArgs();
const uniformGrid = createSurfelHashGrid();
const screenDebug = createSurfelScreenDebug(uniformGrid, surfelPool);

// Create BVH & Pass
const integratorDispatchArgs = createIntegratorDispatchArgs();
let surfelIntegrate: ReturnType<typeof createSurfelIntegratePass> | null = null;

screenDebug.setDebugMode(screenDebug.debugParams.mode);
screenDebug.configureGUI(gui);

const { updateAnimation, updateLightFromAngles, lightCfg, setLight } =
  createLightControls(gui, dirLight);

const GI_MODES = {
  Direct: 'direct',
  Indirect: 'indirect',
  Combined: 'combined',
} as const;
type GiMode = Exclude<NonNullable<SceneSettings['gi']>['mode'], undefined>;
const giParams: { mode: GiMode; indirectIntensity: number } = {
  mode: GI_MODES.Combined,
  indirectIntensity: 1.0,
};

const giFolder = gui.addFolder('GI');
let mustRebuildCompositeMaterial = true;
const giModeController = giFolder
  .add(giParams, 'mode', GI_MODES)
  .name('Output')
  .onChange(() => {
    mustRebuildCompositeMaterial = true;
  });
const giIndirectController = giFolder
  .add(giParams, 'indirectIntensity', 0, 10, 0.1)
  .name('Indirect Intensity')
  .onChange(() => {
    mustRebuildCompositeMaterial = true;
  });
giModeController.listen?.();
giIndirectController.listen?.();

configureRadialDepthGUI(gui);

function applyLightSettings(settings: NonNullable<SceneSettings['light']>) {
  if (settings.azimuthDeg !== undefined)
    lightCfg.azimuthDeg = settings.azimuthDeg;
  if (settings.elevationDeg !== undefined)
    lightCfg.elevationDeg = settings.elevationDeg;
  if (settings.intensity !== undefined) lightCfg.intensity = settings.intensity;
  if (settings.animate !== undefined) lightCfg.animate = settings.animate;
  if (settings.speed !== undefined) lightCfg.speed = settings.speed;
}

function applyGiSettings(settings: NonNullable<SceneSettings['gi']>) {
  if (settings.mode !== undefined) giParams.mode = settings.mode;
  if (settings.indirectIntensity !== undefined)
    giParams.indirectIntensity = settings.indirectIntensity;
  mustRebuildCompositeMaterial = true;
}

function applyCameraSettings(settings?: NonNullable<SceneSettings['camera']>) {
  const next = settings ?? defaultCameraSettings;
  camera.position.copy(next.position);
  const target = next.target ?? defaultCameraSettings.target ?? controls.target;
  controls.target.copy(target);
  controls.update();
}

function applySceneSettings(settings?: SceneSettings) {
  if (settings?.gi) applyGiSettings(settings.gi);
  if (settings?.occlusion) applyOcclusionSettings(settings.occlusion);
  if (settings?.light) applyLightSettings(settings.light);
  applyCameraSettings(settings?.camera);
}

let sceneBVH: SceneBVHBundle | null = null;
let sceneLoadToken = 0;

async function loadScene(sceneDef: SceneDefinition) {
  console.log('Loading', sceneDef.label);
  const loadToken = ++sceneLoadToken;
  setLoading(`Loading ${sceneDef.label}`);

  sceneBVH = null;
  dirLight = recreateDirectionalLight(scene, dirLight, dirLightDefaults);
  clearSceneContent(scene);
  setLight(dirLight);
  surfelPrepare.run(renderer, surfelPool, { forceClear: true });

  try {
    envTex = await loadEnvTexture(sceneDef.hdr);
  } catch (error) {
    if (loadToken === sceneLoadToken) {
      showError(error);
    }
    return;
  }
  if (loadToken !== sceneLoadToken || !envTex) return;

  const suv = findSunPositionWeighted(envTex);
  console.log('Sun found at UV', suv);
  setLightAnglesFromEnvMapSunUVLocation(suv[0], suv[1]);

  applySceneSettings(sceneDef.settings);
  updateLightFromAngles();

  surfelIntegrate = createSurfelIntegratePass(blueNoise, envTex);

  try {
    await sceneDef.populate(scene, dirLight);
  } catch (error) {
    if (loadToken === sceneLoadToken) {
      showError(error);
    }
    return;
  }
  if (loadToken !== sceneLoadToken) return;

  try {
    setLoading('Building BVH');
    sceneBVH = createSceneBVH(renderer, scene);
  } catch (error) {
    if (loadToken === sceneLoadToken) {
      showError(error);
    }
    return;
  }
  if (loadToken === sceneLoadToken) {
    clearLoading();
  }
}

async function loadSceneById(sceneId: string) {
  const nextScene = SCENE_PRESETS.find((scene) => scene.id === sceneId);
  if (!nextScene) return;
  await loadScene(nextScene);
}

const sceneOptions = SCENE_PRESETS.map((scene) => ({
  id: scene.id,
  label: scene.label,
}));
const sceneIdSet = new Set(sceneOptions.map((scene) => scene.id));
const sceneIdByLabel = new Map(
  sceneOptions.map((scene) => [scene.label, scene.id]),
);
const sceneLabelById = new Map(
  sceneOptions.map((scene) => [scene.id, scene.label]),
);
const defaultSceneId = sceneOptions[0]?.id;
let currentSceneId = defaultSceneId;
let isSyncingSceneSelection = false;

function normalizeSceneId(value: string): string {
  if (sceneIdSet.has(value)) return value;
  return sceneIdByLabel.get(value) ?? value;
}

const sceneSwitcher = createSceneSwitcher(
  sceneOptions,
  (sceneId) => {
    if (isSyncingSceneSelection) return;
    selectScene(normalizeSceneId(sceneId));
  },
  defaultSceneId,
);

const inspectorSceneSelector = defaultSceneId
  ? configureSceneSelector(
      gui,
      sceneOptions,
      (sceneId) => {
        if (isSyncingSceneSelection) return;
        selectScene(normalizeSceneId(sceneId));
      },
      defaultSceneId,
    )
  : null;

function syncSceneSelection(sceneId: string) {
  isSyncingSceneSelection = true;
  try {
    currentSceneId = sceneId;
    sceneSwitcher?.setValue(sceneId);
    if (inspectorSceneSelector) {
      inspectorSceneSelector.params.scene = sceneId;
      if ('select' in inspectorSceneSelector.controller) {
        const label = sceneLabelById.get(sceneId);
        if (label) {
          inspectorSceneSelector.controller.select.value = label;
        }
      } else {
        inspectorSceneSelector.controller.updateDisplay?.();
      }
    }
  } finally {
    isSyncingSceneSelection = false;
  }
}

function selectScene(sceneId: string) {
  const normalized = normalizeSceneId(sceneId);
  if (!normalized || !sceneIdSet.has(normalized)) {
    return;
  }
  if (normalized === currentSceneId) {
    syncSceneSelection(normalized);
    return;
  }
  syncSceneSelection(normalized);
  void loadSceneById(normalized);
}

if (defaultSceneId) {
  await loadSceneById(defaultSceneId);
}

const surfelResolve = createSurfelGIResolvePass(uniformGrid, surfelPool);

// --------------------------------------------------------
// COMPOSITING SETUP
// --------------------------------------------------------
const postProcessing = new THREE.PostProcessing(renderer);
const scenePass = pass(scene, camera);

scenePass.setMRT(
  mrt({
    output: output,
    velocity: velocity,
  }),
);

const scenePassColor = scenePass.getTextureNode('output').toInspector('Color');
const scenePassDepth = scenePass
  .getTextureNode('depth')
  .toInspector('Depth', () => {
    return scenePass.getLinearDepthNode();
  });

const scenePassVelocity = scenePass
  .getTextureNode('velocity')
  .toInspector('Velocity');

const traaNode = traa(
  scenePassColor,
  scenePassDepth,
  scenePassVelocity,
  camera,
);
postProcessing.outputNode = traaNode;

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  gbuffer.resize(renderer);
});

const prevCameraPos = new THREE.Vector3();
prevCameraPos.copy(camera.position);

renderer.setAnimationLoop(() => {
  if (renderer.info.frame < 100) {
    const csmHelper = scene.userData.csmHelper;
    if (csmHelper) {
      csmHelper.update();
      csmHelper.updateVisibility();
    }
  }

  controls.update();
  updateAnimation();
  camera.updateMatrixWorld();

  if (!sceneBVH || !surfelIntegrate) {
    // postProcessing.render();       // show direct scene while loading
    prevCameraPos.copy(camera.position);
    return;
  }
  scene.background = null;
  // Offscreen gbuffer for spawning
  const prevTarget = renderer.getRenderTarget();
  camera.layers.set(0);
  renderer.setMRT(gbuffer.sceneMRT);
  renderer.setRenderTarget(gbuffer.target);

  renderer.render(scene, camera);
  renderer.setRenderTarget(prevTarget);
  renderer.setMRT(null);
  camera.layers.enable(1);

  // GPU surfel prepare + generation into a fixed-capacity pool (capacity set once at init)
  // Age and recycle a fraction of the pool; this replenishes the free list gradually
  // Rebuild the alive list each frame (compaction): returns indices of currently alive surfels
  surfelPrepare.run(renderer, surfelPool);

  const findResult = surfelFindMissing.run(
    renderer,
    camera,
    gbuffer,
    surfelPool,
    uniformGrid,
    prevCameraPos,
  );
  surfelDispatchArgs.run(renderer, surfelPool);
  const indirectAttr = surfelDispatchArgs.getIndirectAttr();

  surfelAge.run(
    renderer,
    surfelPool,
    surfelFindMissing,
    uniformGrid,
    prevCameraPos,
    indirectAttr,
  );
  surfelAllocate.run(
    renderer,
    surfelPool,
    surfelFindMissing,
    findResult.tileCount,
  );

  // Rebuild grid to include freshly spawned surfels for downstream passes
  uniformGrid.build(renderer, surfelPool, camera);

  // Get the dispatch args for the integrator
  integratorDispatchArgs.run(renderer, surfelPool);
  // 2. INTEGRATE (Ray Trace)
  surfelIntegrate.run(
    renderer,
    surfelPool,
    sceneBVH,
    uniformGrid,
    camera,
    dirLight,
    integratorDispatchArgs.getIndirectAttr(),
  );

  // 3. Resolve GI (Compute Indirect Light Texture)
  surfelResolve.run(renderer, camera, gbuffer);

  // Surfel health debug
  // debugSurfelSystem(renderer, surfelPool, uniformGrid, surfelFindMissing);
  // debugOffsetsStats(renderer, uniformGrid)
  // build the per-pixel overlay from GBuffer + CSR
  const debugActive = screenDebug.debugParams.mode !== SCREEN_DEBUG_MODES.Off;

  // 5. Composite Final Image (Fullscreen Pass)
  const giTex = surfelResolve.getOutputTexture();
  let directLight;
  if (giTex) {
    if (mustRebuildCompositeMaterial) {
      directLight = scenePassColor; // Direct Light
      const albedo = texture(gbuffer.target.textures[1], screenUV);
      const indirectLight = texture(giTex, screenUV)
        .mul(albedo)
        .mul(giParams.indirectIntensity);

      switch (giParams.mode) {
        case GI_MODES.Direct:
          postProcessing.outputNode = fxaa(directLight);
          // postProcessing.outputNode = traa( directLight, scenePassDepth, scenePassVelocity, camera );
          postProcessing.needsUpdate = true;
          break;
        case GI_MODES.Indirect:
          postProcessing.outputNode = fxaa(indirectLight);
          postProcessing.needsUpdate = true;
          // postProcessing.outputNode = traa( indirectLight, scenePassDepth, scenePassVelocity, camera );
          break;
        case GI_MODES.Combined:
        default:
          // postProcessing.outputNode = traa(directLight.add(indirectLight), scenePassDepth, scenePassVelocity, camera);
          postProcessing.outputNode = fxaa(directLight.add(indirectLight)); //, scenePassDepth, scenePassVelocity, camera );
          postProcessing.needsUpdate = true;
          break;
      }
      mustRebuildCompositeMaterial = false;
    }
  }

  scene.background = envTex;
  postProcessing.render();

  if (debugActive) {
    screenDebug.run(
      renderer,
      camera,
      gbuffer,
      surfelFindMissing,
      prevCameraPos,
    );
    renderer.render(scene, camera);
    screenDebug.renderOverlay(renderer);
  }

  // Update prevCameraPos for the next frame
  prevCameraPos.copy(camera.position);
  surfelPool.swapMoments();
});
