import { GUI } from 'lil-gui';
import type * as THREE from 'three/webgpu';
import type {
  Inspector,
  InspectorParametersGroup,
} from 'three/addons/inspector/Inspector.js';

type GuiFolder = {
  addFolder: (label: string) => GuiFolder;
  add: (object: any, property: string, ...params: any[]) => GuiController;
  close?: () => GuiFolder | void;
  open?: (open?: boolean) => GuiFolder;
};

type GuiController = {
  name: (label: string) => GuiController;
  onChange: (callback: (value: any) => void) => GuiController;
  listen?: () => GuiController | void;
  updateDisplay?: () => GuiController | void;
};

function isInspectorAvailable(
  renderer: THREE.WebGPURenderer,
): renderer is THREE.WebGPURenderer & {
  inspector: Inspector & {
    createParameters: (label: string) => InspectorParametersGroup;
  };
} {
  return Boolean(
    renderer.inspector &&
    typeof (renderer.inspector as Inspector).createParameters === 'function',
  );
}

export function createUI(renderer: THREE.WebGPURenderer): GuiFolder {
  if (isInspectorAvailable(renderer)) {
    const inspector = renderer.inspector as Inspector & {
      profiler?: {
        miniPanel?: HTMLElement;
        tabs?: { parameters?: any; performance?: { id?: string } };
        setActiveTab?: (id: string) => void;
      };
    };

    const controls = inspector.createParameters('Controls');

    const closeMiniParams = () => {
      const profiler = inspector.profiler;
      const paramsTab = profiler?.tabs?.parameters;
      if (!profiler || !paramsTab) return;

      if (paramsTab.miniContent?.firstChild) {
        paramsTab.content.appendChild(paramsTab.miniContent.firstChild);
      }
      if (paramsTab.miniContent) {
        paramsTab.miniContent.style.display = 'none';
      }
      if (paramsTab.builtinButton) {
        paramsTab.builtinButton.classList.remove('active');
      }
      if (profiler.miniPanel) {
        profiler.miniPanel.classList.remove('visible');
      }

      const performanceId = profiler.tabs?.performance?.id;
      if (performanceId && profiler.setActiveTab) {
        profiler.setActiveTab(performanceId);
      }
    };

    closeMiniParams();
    requestAnimationFrame(closeMiniParams);

    const toggleIcon = document.getElementById('toggle-icon');
    if (toggleIcon) {
      toggleIcon.style.display = 'none';
    }

    return controls;
  }

  return new GUI();
}

export type SceneOption = {
  id: string;
  label: string;
};

export type SceneSwitcher = {
  root: HTMLDivElement;
  select: HTMLSelectElement;
  nextButton: HTMLButtonElement;
  setValue: (sceneId: string) => void;
};

export function createSceneSwitcher(
  scenes: SceneOption[],
  onChange: (sceneId: string) => void,
  initialSceneId?: string,
): SceneSwitcher | null {
  if (scenes.length === 0) return null;

  const root = document.createElement('div');
  root.id = 'scene-switcher';

  const select = document.createElement('select');
  select.id = 'scene-select';
  for (const scene of scenes) {
    const option = document.createElement('option');
    option.value = scene.id;
    option.textContent = scene.label;
    select.appendChild(option);
  }
  select.value = initialSceneId ?? scenes[0].id;
  select.addEventListener('change', () => onChange(select.value));

  const nextButton = document.createElement('button');
  nextButton.id = 'scene-next';
  nextButton.type = 'button';
  nextButton.textContent = 'Next';
  nextButton.addEventListener('click', () => {
    const currentIndex = scenes.findIndex((scene) => scene.id === select.value);
    const nextIndex =
      currentIndex === -1 ? 0 : (currentIndex + 1) % scenes.length;
    onChange(scenes[nextIndex].id);
  });

  root.append(select, nextButton);
  document.body.appendChild(root);

  return {
    root,
    select,
    nextButton,
    setValue: (sceneId: string) => {
      select.value = sceneId;
    },
  };
}

export function configureSceneSelector(
  gui: GuiFolder,
  scenes: SceneOption[],
  onChange: (sceneId: string) => void,
  initialSceneId?: string,
) {
  if (scenes.length === 0) return;

  const sceneParams = {
    scene: initialSceneId ?? scenes[0].id,
  };

  const options = Object.fromEntries(
    scenes.map((scene) => [scene.label, scene.id]),
  );
  const folder = gui.addFolder('Scene');
  const controller = folder
    .add(sceneParams, 'scene', options)
    .name('Preset')
    .onChange((value) => {
      onChange(String(value));
    });

  controller.listen?.();
  return { folder, controller, params: sceneParams };
}
