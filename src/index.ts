// @ts-ignore
import { Object as OlObj, Map } from 'ol';
// @ts-ignore
import { listen, unlistenByKey } from 'ol/events';

function removeNode(node: HTMLElement) {
  return node && node.parentNode ? node.parentNode.removeChild(node) : null;
}

const _options = {
  forcedRerender: false, // Force re-rendering
  forcedPrecomposeRerender: false, // force pre re-render
  hideOnZooming: false, // when zooming hide layer
  hideOnMoving: false, // when moving hide layer
  hideOnRotating: false, // // when Rotating hide layer
};

class OverlayLayer extends OlObj {
  public element: HTMLElement | undefined;
  public $options: object;
  private _size: number[] | undefined;
  private readonly id: any;
  private readonly insertFirst: Boolean | undefined;
  private initedEvent: Boolean;
  private visible: Boolean;
  constructor(options: object) {
    super(options);

    this.$options = Object.assign(_options, options);

    // @ts-ignore
    this.id = options.id;

    /**
     * size
     */
    this._size = undefined;

    /**
     * event has inited
     */
    this.initedEvent = false;

    /**
     * layer visible
     */
    this.visible = true;

    // @ts-ignore
    this.insertFirst = options.insertFirst !== undefined ? options.insertFirst : true;

    this.render = this.render.bind(this);
    this.checkUpdate = this.checkUpdate.bind(this);
    this.createLayerContainer();

    listen(this, 'change:map', this.handleMapChanged, this);
  }

  getElement(): HTMLElement | undefined {
    return this.element;
  }

  getId(): any {
    return this.id;
  }

  setMap(map: any): void {
    this.set('map', map);
  }

  getMap() {
    return this.get('map');
  }

  /**
   * create container
   */
  createLayerContainer () {
    const container = (this.element = document.createElement('div'));
    container.style.position = 'absolute';
    container.style.top = '0px';
    container.style.left = '0px';
    container.style.right = '0px';
    container.style.bottom = '0px';
  }

  /**
   * handle map changed
   */
  handleMapChanged () {
    if (this.initedEvent) {
      this.element && removeNode(this.element);
      this.unBindEvent();
    }
    const map = this.getMap();
    if (map) {
      this.bindEvent(map);
      const container = map.getOverlayContainer();
      if (this.insertFirst) {
        container.insertBefore(this.element, container.childNodes[0] || null);
      } else {
        container.appendChild(this.element);
      }
    }
  }

  /**
   * bind map update
   */
  render(): void {
    this.dispatchEvent({
      type: 'render',
      value: this,
    });
  }

  prerender(): void {
    this.dispatchEvent({
      type: 'prerender',
      value: this,
    });
  }

  /**
   * update view size
   * @param size
   */
  updateViewSize(size: number[]): void {
    if (!this.element) return;
    this._size = size;
    this.element.style.width = `${size[0]}px`;
    this.element.style.height = `${size[1]}px`;
    this.element.setAttribute('width', String(size[0]));
    this.element.setAttribute('height', String(size[1]));
  }

  /**
   * update layer
   */
  checkUpdate() {
    const map = this.getMap();
    if (!map || !map.isRendered()) {
      this.setVisible(false);
      return;
    }

    const mapSize = map.getSize();
    if (!this._size) {
      this.updateViewSize(mapSize);
    } else {
      const _update = this._size[0] === 0 || this._size[1] === 0
        || (this._size[0] !== mapSize[0] || this._size[1] !== mapSize[1]);
      _update && this.updateViewSize(mapSize);
    }
  }

  setVisible(visible: Boolean): void {
    if (this.visible !== visible) {
      // @ts-ignore
      this.element.style.display = visible ? '' : 'none';
      this.visible = visible;
    }
    this.dispatchEvent({
      type: 'change:visible ',
      value: visible,
    });
  }

  /**
   * is visible
   * @returns {Element|*|boolean}
   * @private
   */
  isVisible (): Boolean {
    return this.visible;
  }

  dispatchEvent(...args: any) {
    super.dispatchEvent(...args);
  }

  show() {
    this.setVisible(true);
  }

  hide() {
    this.setVisible(false);
  }

  private bindEvent(map: any) {
    // const view = map.getView();
    // map.on('rendercomplete', this.render);
    map.on('change:size', this.checkUpdate);
    // view.on('change:center', this.render);
    // view.on('change:resolution', this.render);
    // view.on('change:rotation', this.render);
    // map.on('postcompose', this.render);
    // map.on('postrender', this.render_);
    map.on('precompose', this.render);
    // map.on('movestart', this.render);
    // map.on('moveend', this.render);
    this.initedEvent = true;
  }

  private unBindEvent() {
    const map = this.getMap();
    // map.un('rendercomplete', this.render);
    // map.un('postcompose', this.render);
    // map.un('postrender', this.render);
    map.un('precompose', this.render);
    map.un('change:size', this.checkUpdate);
    this.initedEvent = false;
  }

  private set(key: string, value: any) {
    return super.set(key, value);
  }

  private get(key: string) {
    return super.get(key);
  }
}

export default OverlayLayer;
