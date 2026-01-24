import type InspectorBase from 'three/src/renderers/common/InspectorBase.js';

declare module 'three/addons/inspector/Inspector.js' {
  export type InspectorParameterControl = {
    name: (label: string) => InspectorParameterControl;
    onChange: (callback: (value: any) => void) => InspectorParameterControl;
    listen: () => InspectorParameterControl;
  };

  export type InspectorParametersGroup = {
    add: (
      object: any,
      property: string,
      ...params: any[]
    ) => InspectorParameterControl;
    addFolder: (name: string) => InspectorParametersGroup;
    close: () => InspectorParametersGroup;
  };

  export class Inspector extends InspectorBase {
    constructor();
    domElement: HTMLElement;
    profiler: unknown;
    console: unknown;
    performance: unknown;
    parameters: unknown;
    createParameters(name: string): InspectorParametersGroup;
  }
}
