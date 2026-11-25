import {
  ALL_LEGEND_KEYS,
  legendKeyOf,
  hash01,
  METRES_PER_DEG_LAT,
} from './geo';
import type {
  FacilityCard,
  Footprint,
  Graph,
  GraphEdge,
  GraphNode,
  RouteOption,
} from '../types';

const SVG_NS = 'http://www.w3.org/2000/svg';

/* A real street map underneath, so the campus sits on roads instead of a blank
   page. Nothing is bundled: the browser fetches the few tiles the view needs, and
   offline the illustration underneath still works. */
const TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const MAX_TILES = 42;

/* Marker sizes in map units (metres on the ground), so they scale with the map like
   the buildings do. Halved from the original illustration: at 18 units a building pin
   covered most of the footprint it was pointing at. */
/* A place pin, in map units (metres on the ground). Facilities are circles of
   SERVICE_RADIUS above; blocks are footprints, not pins. */
const PLACE_SIZE = 7;
const SERVICE_RADIUS = 3.5;

export interface ScaleInfo {
  width: number;
  label: string;
}

interface MapOptions {
  onScale(info: ScaleInfo): void;
  onPlaceClick(node: GraphNode): void;
  onTip(text: string | null): void;
}

interface Bounds {
  minLat: number;
  maxLat: number;
  minLng: number;
  maxLng: number;
  cos: number;
}

interface View {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Point {
  x: number;
  y: number;
}

/* A published name usually carries its building in brackets: "Block V (Jockey Club
   Innovation Tower)". On the map the block letter is already printed on the roof, so the
   bracketed half is noise -- and at route zoom it ran across half the screen. */
function mapLabel(name: string): string {
  const bare = name.replace(/\s*\([^()]*\)\s*$/, '').trim();
  const text = bare || name;
  return text.length > 26 ? text.slice(0, 25).trimEnd() + '\u2026' : text;
}

function el(name: string, attrs: Record<string, string | number>): SVGElement {
  const node = document.createElementNS(SVG_NS, name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function layerOf(id: string): SVGGElement | null {
  return document.getElementById(id) as SVGGElement | null;
}

/**
 * Everything the campus map draws. The drawing stays imperative on purpose: it is
 * a few hundred SVG elements rebuilt from data, and React re-rendering them on every
 * pan frame would be slower and no clearer. React owns the chrome around it.
 */
export class CampusMap {
  private readonly svg: SVGSVGElement;
  private readonly opts: MapOptions;
  private graph: Graph | null = null;
  private nodes = new Map<string, GraphNode>();
  private bounds: Bounds | null = null;
  private view: View = { x: 0, y: 0, w: 800, h: 600 };
  private basemap = true;
  private showLinks = false;
  private hidden = new Set<string>(ALL_LEGEND_KEYS);
  private serviceGroups: { el: SVGGElement; cat: string }[] = [];
  private drag: { x: number; y: number; vx: number; vy: number } | null = null;
  private teardown: (() => void)[] = [];
  private routes: RouteOption[] = [];
  private selectedRoute = 0;
  private facilities: FacilityCard[] = [];

  constructor(svg: SVGSVGElement, opts: MapOptions) {
    this.svg = svg;
    this.opts = opts;
    this.svg.classList.toggle('basemap', this.basemap);
    this.svg.classList.toggle('no-links', !this.showLinks);
    this.bindInteractions();
  }

  dispose(): void {
    this.teardown.forEach((fn) => fn());
    this.teardown = [];
  }

  // -- data in -------------------------------------------------------------
  setGraph(graph: Graph): void {
    this.graph = graph;
    this.nodes = new Map(graph.nodes.map((node) => [node.id, node]));
    this.setupProjection();
    this.drawGraph();
    // Routes and facilities can arrive *before* the graph does -- the browser restores a
    // saved conversation on load, and its route is pushed while the /api/graph request is
    // still in flight. They were drawn then with no projection and threw on the first
    // coordinate. Redrawing here makes the order of those two requests irrelevant.
    if (this.routes.length) this.setRoutes(this.routes, this.selectedRoute);
    if (this.facilities.length) this.renderFacilities();
  }

  setHiddenCategories(hidden: Set<string>): void {
    this.hidden = new Set(hidden);
    this.applyFilters();
  }

  setBasemap(on: boolean): void {
    this.basemap = on;
    this.svg.classList.toggle('basemap', on);
    this.drawTiles();
    this.drawFootprints();
  }

  setShowLinks(on: boolean): void {
    this.showLinks = on;
    this.svg.classList.toggle('no-links', !on);
  }

  /**
   * Draw every candidate route at once, one colour each, with the chosen one on top
   * and at full strength. Only the chosen route gets dots and START/GOAL labels:
   * three sets of pins on one line is unreadable.
   */
  setRoutes(routes: RouteOption[], selected: number): void {
    this.routes = routes;
    this.selectedRoute = selected;
    // Framing the route projects its coordinates, so this waits for the graph too; the
    // request is remembered and applied by setGraph().
    if (!this.bounds) return;
    this.renderRoutes();
    const chosen = routes[Math.max(0, Math.min(selected, routes.length - 1))];
    if (chosen) this.fitTo(this.routeExtent(chosen));
  }

  /**
   * Everything the chosen route puts on screen: every point of the drawn line, plus its
   * landmarks. Fitting to the two endpoints alone framed a straight line through the
   * buildings and let a walk that bends around them run off the edge -- which is also why
   * the labels looked enormous: the view was zoomed further in than the route.
   */
  private routeExtent(route: RouteOption): Point[] {
    const points: Point[] = [];
    (route.segments || []).forEach((segment) => {
      (segment.path || []).forEach(([lat, lng]) => points.push(this.project(lat, lng)));
    });
    (route.landmarks || []).forEach((mark) => points.push(this.project(mark.lat, mark.lng)));
    if (!points.length) {
      (route.path || []).forEach((p) => points.push(this.project(p.lat, p.lng)));
    }
    return points;
  }

  /** The point "metres" along a projected line. Map units are metres on the ground. */
  private pointAlong(points: Point[], metres: number): Point {
    if (!points.length) return { x: 0, y: 0 };
    let left = Math.max(0, metres);
    for (let index = 1; index < points.length; index += 1) {
      const from = points[index - 1];
      const to = points[index];
      const length = Math.hypot(to.x - from.x, to.y - from.y);
      if (length <= 0) continue;
      if (left <= length) {
        const t = left / length;
        return { x: from.x + (to.x - from.x) * t, y: from.y + (to.y - from.y) * t };
      }
      left -= length;
    }
    return points[points.length - 1];
  }

  /** A label that stays on the canvas: near the right edge it flips to the other side. */
  private routeLabel(x: number, y: number, text: string, extra = ''): SVGElement {
    const label = el('text', { x: x + 6, y, class: 'node-label route-label ' + extra });
    label.textContent = text;
    // about 5.6 map units per character: an estimate, used only to decide the flip
    if (x + 6 + text.length * 5.6 > this.view.x + this.view.w) {
      label.setAttribute('x', String(x - 6));
      label.setAttribute('text-anchor', 'end');
    }
    return label;
  }

  /** The route layer alone: called again when only the zoom (and so the pin size) moved. */
  private renderRoutes(): void {
    const layer = layerOf('layer-route');
    // No projection yet means no coordinates yet: the graph has not arrived. setGraph()
    // redraws whatever is waiting as soon as it has one.
    if (!layer || !this.bounds) return;
    layer.innerHTML = '';
    this.svg.classList.toggle('no-links', !this.showLinks);
    const routes = this.routes;
    if (!routes.length) return;

    const chosen = Math.max(0, Math.min(this.selectedRoute, routes.length - 1));
    // draw the others first so the chosen line is never buried under them
    const order = routes
      .map((route, index) => ({ route, index }))
      .sort((a, b) => (a.index === chosen ? 1 : 0) - (b.index === chosen ? 1 : 0));

    order.forEach(({ route, index }) => {
      const isChosen = index === chosen;
      const dots = route.path.map((p) => this.project(p.lat, p.lng));
      // the line follows the walkway geometry the engine measured, not the dots
      const shaped: Point[] = [];
      route.segments.forEach((segment) => {
        if (segment.path && segment.path.length >= 2) {
          segment.path.forEach(([lat, lng]) => shaped.push(this.project(lat, lng)));
        }
      });
      const points = shaped.length >= 2 ? shaped : dots;
      const d = points
        .map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(2) + ' ' + p.y.toFixed(2))
        .join(' ');
      if (!d) return;

      const colour = `var(--route-${index + 1})`;
      const casing = el('path', { d, class: 'route-casing' });
      const line = el('path', { d, class: 'route-line' });
      if (isChosen) {
        line.setAttribute('style', `stroke: ${colour}`);
      } else {
        casing.setAttribute('style', 'stroke-width: 3.5; opacity: .55');
        line.setAttribute(
          'style',
          `stroke: ${colour}; stroke-width: 2; opacity: .55`,
        );
      }
      layer.appendChild(casing);
      layer.appendChild(line);

      if (!isChosen) return;
      // The waypoints are bends in a walkway, so the line itself carries no names. The
      // buildings it passes do, and they are what a person recognises on the way, so the
      // chosen route is annotated with them -- at the point on the walk where you pass
      // them. The engine measured that distance already (at_m), and the drawn line is in
      // metres, so the dot lands on the path instead of on the building's published
      // centre, which can be twenty metres away and left the dots floating beside it.
      const marks: { x: number; y: number; name: string }[] = [];
      (route.landmarks || []).forEach((mark) => {
        const at = this.pointAlong(points, mark.at_m);
        // Two POIs inside one building share a published coordinate and so a distance
        // along the walk ("Block V" and "Innovation Gallery" are the same point). One dot
        // and one name is what the map can honestly show.
        if (marks.some((other) => Math.hypot(other.x - at.x, other.y - at.y) < 8)) return;
        marks.push({ x: at.x, y: at.y, name: mapLabel(mark.name) });
      });
      marks.forEach((mark, index) => {
        // In map units, like every other mark: 3.4 is about 7 m across.
        const dot = el('circle', { cx: mark.x, cy: mark.y, r: 3.4, class: 'route-landmark-dot' });
        // In the line's own colour. The stylesheet's default is route 1, so a landmark on
        // the second or third option used to stay navy -- a dot in the wrong colour reads as
        // belonging to the wrong walk, and the whole point of the annotation is which walk
        // it belongs to.
        dot.setAttribute('style', 'fill: ' + colour);
        layer.appendChild(dot);
        // alternate above and below the line, so neighbouring names do not stack up
        const side = index % 2 === 0 ? -7 : 12;
        const label = this.routeLabel(mark.x, mark.y + side, mark.name, 'route-landmark');
        label.setAttribute('style', 'fill: ' + colour);
        layer.appendChild(label);
      });
      route.path.forEach((point, i) => {
        const xy = dots[i];
        // The stops and the ends carry the route's colour too, for the same reason.
        const ring = el('circle', { cx: xy.x, cy: xy.y, r: i === 0 ? 4.5 : 3, class: 'route-dot' });
        ring.setAttribute('style', 'stroke: ' + colour);
        layer.appendChild(ring);
        const pin = el('circle', { cx: xy.x, cy: xy.y, r: i === 0 ? 2 : 1.5, class: 'route-pin' });
        pin.setAttribute('style', 'fill: ' + colour);
        layer.appendChild(pin);
        if (i === 0 || i === route.path.length - 1) {
          // the short name: the full published one is a third of the screen at this zoom
          layer.appendChild(
            this.routeLabel(xy.x, xy.y - 6,
              (i === 0 ? 'START · ' : 'GOAL · ') + mapLabel(point.name)),
          );
        }
      });
    });
  }

  setFacilities(facilities: FacilityCard[]): void {
    this.facilities = facilities;
    this.renderFacilities();
  }

  private renderFacilities(): void {
    const layer = layerOf('layer-facilities');
    if (!layer || !this.bounds) return;
    layer.innerHTML = '';
    this.facilities.forEach((facility) => {
      if (facility.lat == null || facility.lng == null) return;
      const p = this.project(facility.lat, facility.lng);
      layer.appendChild(
        el('circle', { cx: p.x, cy: p.y, r: 13, class: 'fac-pin pulse' }),
      );
    });
  }

  clearOverlays(): void {
    const route = layerOf('layer-route');
    const facilities = layerOf('layer-facilities');
    if (route) route.innerHTML = '';
    if (facilities) facilities.innerHTML = '';
    this.svg.classList.toggle('no-links', !this.showLinks);
  }

  /** Pan and zoom so one place fills the view, as a map app does from its search box. */
  focusPlace(node: GraphNode): void {
    const p = this.project(node.lat, node.lng);
    const span = 420;
    this.view = { x: p.x - span / 2, y: p.y - span / 2, w: span, h: span * 0.72 };
    this.applyView();
  }

  fitAll(): void {
    if (!this.graph) return;
    this.setupProjection();
    this.setRoutes([], 0);
  }

  zoom(factor: number, cx?: number, cy?: number): void {
    const v = this.view;
    const b = this.bounds;
    if (!b) return;
    const limit = (b.maxLng - b.minLng) * METRES_PER_DEG_LAT * b.cos * 3;
    const nw = Math.min(Math.max(v.w * factor, 120), limit);
    const nh = nw * (v.h / v.w);
    const px = cx == null ? v.x + v.w / 2 : cx;
    const py = cy == null ? v.y + v.h / 2 : cy;
    v.x = px - (px - v.x) * (nw / v.w);
    v.y = py - (py - v.y) * (nh / v.h);
    v.w = nw;
    v.h = nh;
    this.applyView();
  }

  // -- projection ----------------------------------------------------------
  private setupProjection(): void {
    const nodes = this.graph ? this.graph.nodes : [];
    const shapes = (this.graph && this.graph.footprints) || [];
    // Frame the campus by what is drawn on it, not by every node. A handful of places
    // PolyU publishes sit 1.6 km away (the West Kowloon campus, the halls), and letting
    // them set the bounds shrank the buildings to a smudge in the corner. The structures
    // are the campus; nodes only frame the view when there are no structures at all.
    const ring = shapes.flatMap((item) => item.ring);
    if (!nodes.length && !ring.length) return;
    const lats = ring.length ? ring.map(([lat]) => lat) : nodes.map((n) => n.lat);
    const lngs = ring.length ? ring.map(([, lng]) => lng) : nodes.map((n) => n.lng);
    const minLat = Math.min(...lats) - 0.0006;
    const maxLat = Math.max(...lats) + 0.0006;
    const minLng = Math.min(...lngs) - 0.0006;
    const maxLng = Math.max(...lngs) + 0.0006;
    const cos = Math.cos((((minLat + maxLat) / 2) * Math.PI) / 180);
    this.bounds = { minLat, maxLat, minLng, maxLng, cos };
    this.view = {
      x: 0,
      y: 0,
      w: (maxLng - minLng) * METRES_PER_DEG_LAT * cos,
      h: (maxLat - minLat) * METRES_PER_DEG_LAT,
    };
    this.applyView();
  }

  private project(lat: number, lng: number): Point {
    const b = this.bounds as Bounds;
    return {
      x: (lng - b.minLng) * METRES_PER_DEG_LAT * b.cos,
      y: (b.maxLat - lat) * METRES_PER_DEG_LAT,
    };
  }

  private unproject(x: number, y: number): { lat: number; lng: number } {
    const b = this.bounds as Bounds;
    return {
      lng: b.minLng + x / (METRES_PER_DEG_LAT * b.cos),
      lat: b.maxLat - y / METRES_PER_DEG_LAT,
    };
  }

  private applyView(): void {
    const v = this.view;
    this.svg.setAttribute('viewBox', v.x + ' ' + v.y + ' ' + v.w + ' ' + v.h);
    // Names are expensive: a label appears only when there is room for it AND it is
    // the kind of place worth naming at that zoom. Route labels are exempt.
    const roofsAreLabelled = this.svg.classList.contains('footprints');
    document.querySelectorAll<SVGTextElement>('.node-label:not(.route-label)').forEach((label) => {
      const wanted = label.classList.contains('building-label')
        ? v.w < 1400 && !roofsAreLabelled
        : label.classList.contains('always');
      label.style.display = wanted ? '' : 'none';
    });
    this.updateScale();
    this.drawTiles();
  }

  /**
   * A scale bar turned from real metres: the view is metres wide, so pixels per
   * metre is just the panel width divided by the view width.
   */
  private updateScale(): void {
    if (!this.bounds) return;
    const rect = this.svg.getBoundingClientRect();
    if (!rect.width) return;
    const metresPerPx = this.view.w / rect.width;
    const target = metresPerPx * 72;
    const pow = Math.pow(10, Math.floor(Math.log10(target)));
    const nice = [1, 2, 5, 10].map((m) => m * pow).find((value) => value >= target) || target;
    this.opts.onScale({
      width: Number((nice / metresPerPx).toFixed(1)),
      label: nice >= 1000 ? (nice / 1000).toFixed(nice % 1000 ? 1 : 0) + ' km' : Math.round(nice) + ' m',
    });
  }

  // -- basemap tiles -------------------------------------------------------
  private tileZoom(): number {
    const rect = this.svg.getBoundingClientRect();
    if (!rect.width || !this.bounds) return 17;
    const metresPerPixel = this.view.w / rect.width;
    const lat = (this.bounds.minLat + this.bounds.maxLat) / 2;
    const worldMetres = 156543.03392 * Math.cos((lat * Math.PI) / 180);
    return Math.max(15, Math.min(19, Math.round(Math.log2(worldMetres / metresPerPixel))));
  }

  private drawTiles(): void {
    const layer = layerOf('layer-tiles');
    if (!layer) return;
    layer.innerHTML = '';
    if (!this.basemap || !this.bounds) return;

    const z = this.tileZoom();
    const nw = this.unproject(this.view.x, this.view.y);
    const se = this.unproject(this.view.x + this.view.w, this.view.y + this.view.h);
    const x0 = Math.floor(tileXOf(nw.lng, z));
    const x1 = Math.floor(tileXOf(se.lng, z));
    const y0 = Math.floor(tileYOf(nw.lat, z));
    const y1 = Math.floor(tileYOf(se.lat, z));
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > MAX_TILES) return; // too far out: skip

    for (let x = x0; x <= x1; x += 1) {
      for (let y = y0; y <= y1; y += 1) {
        const top = latAtTileY(y, z);
        const bottom = latAtTileY(y + 1, z);
        const left = lngAtTileX(x, z);
        const right = lngAtTileX(x + 1, z);
        const a = this.project(top, left);
        const b = this.project(bottom, right);
        layer.appendChild(
          el('image', {
            href: TILE_URL.replace('{z}', String(z))
              .replace('{x}', String(x))
              .replace('{y}', String(y)),
            x: a.x,
            y: a.y,
            width: b.x - a.x,
            height: b.y - a.y,
            preserveAspectRatio: 'none',
            class: 'tile',
          }),
        );
      }
    }
  }

  // -- 2.5D building blocks ------------------------------------------------
  /* Take each building's real footprint, copy it straight up the screen by a height,
     and fill the ribbon between the two copies -- that ribbon is the wall, the lifted
     copy is the roof. Walls first, then roofs, sorted north to south. */
  private drawFootprints(): void {
    const layer = layerOf('layer-footprints');
    if (!layer || !this.graph) return;
    layer.innerHTML = '';
    const all = this.graph.footprints || [];
    const active = all.length > 0;
    this.svg.classList.toggle('footprints', active);
    if (!active) return;

    const maxHeight =
      parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--fp-height')) || 20;

    // drawn paths first, so blocks sit on top of them
    all
      .filter((item) => item.line)
      .forEach((item) => {
        const pts = item.ring.map(([lat, lng]) => this.project(lat, lng));
        const d = pts.map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(1) + ' ' + p.y.toFixed(1)).join(' ');
        layer.appendChild(el('path', { d, class: 'fp-path fp-path-' + (item.kind || 'path') }));
      });

    const rings: Footprint[] = all.filter((item) => !item.line);

    const blocks = rings
      .map((item) => {
        const coords = item.ring.map(([lat, lng]) => this.project(lat, lng));
        const north = Math.max(...item.ring.map((p) => p[0]));
        // Which letter belongs on this roof? The one the data gives: "display" if
        // present, else the block code behind it. Nothing else. Deriving a code from
        // the name is what put "LE" on the Lee Shau Kee Building (whose block is Y) and
        // "JO" on the Innovation Tower (whose block is V), and it invented codes for the
        // eleven unnamed slabs too.
        const badge = item.display || '';
        // a hand-drawn shape may carry its own height; otherwise vary it deterministically
        const height = item.height || maxHeight * (0.55 + hash01(item.id, 3) * 0.45);
        const tone = item.tone ? ' tone-' + item.tone : '';
        return {
          coords,
          north,
          height,
          tone,
          badge,
          kind: item.kind || 'building',
        };
      })
      .sort((a, b) => b.north - a.north); // north first, south last on top

    blocks.forEach((block) => {
      const { coords, height } = block;

      // a flat feature (pool, lawn, court) is painted once, with no walls and no height
      if (block.kind && block.kind !== 'building') {
        const flat = coords.map((p) => p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ');
        layer.appendChild(el('polygon', { points: flat, class: 'fp-flat fp-flat-' + block.kind }));
        return;
      }

      // 1. a soft shadow on the ground, offset down-right: grounds the block
      const shadow = coords.map((p) => (p.x + 5).toFixed(1) + ',' + (p.y + 5).toFixed(1)).join(' ');
      layer.appendChild(el('polygon', { points: shadow, class: 'fp-shadow' }));

      // 2. the walls
      for (let i = 0; i < coords.length; i += 1) {
        const p1 = coords[i];
        const p2 = coords[(i + 1) % coords.length];
        const d =
          'M' + p1.x.toFixed(1) + ' ' + p1.y.toFixed(1) +
          'L' + p2.x.toFixed(1) + ' ' + p2.y.toFixed(1) +
          'L' + p2.x.toFixed(1) + ' ' + (p2.y - height).toFixed(1) +
          'L' + p1.x.toFixed(1) + ' ' + (p1.y - height).toFixed(1) + 'Z';
        const facing = p2.x < p1.x ? 'fp-wall-dark' : 'fp-wall';
        layer.appendChild(el('path', { d, class: facing + block.tone }));
      }
      const roof = coords.map((p) => p.x.toFixed(1) + ',' + (p.y - height).toFixed(1)).join(' ');
      layer.appendChild(el('polygon', { points: roof, class: 'fp-roof' + block.tone }));

      // 3. a parapet: the roof inset towards its own centre
      const cx = coords.reduce((a, p) => a + p.x, 0) / coords.length;
      const cy = coords.reduce((a, p) => a + p.y, 0) / coords.length;
      const inner = coords
        .map(
          (p) =>
            (cx + (p.x - cx) * 0.82).toFixed(1) +
            ',' +
            (cy + (p.y - cy) * 0.82 - height).toFixed(1),
        )
        .join(' ');
      layer.appendChild(el('polygon', { points: inner, class: 'fp-roof-inner' + block.tone }));

      // 4. the building's letter, on the roof it actually belongs to
      if (block.badge) {
        const code = el('text', {
          x: cx.toFixed(1),
          y: (cy - height + 4).toFixed(1),
          class: 'roof-code',
        });
        code.textContent = block.badge;
        layer.appendChild(code);
      }
    });
  }

  // -- places and walkways -------------------------------------------------
  private drawGraph(): void {
    this.drawEdges();
    this.drawNodes();
    this.drawFootprints();
    this.applyView();
  }

  private drawEdges(): void {
    const edgeLayer = layerOf('layer-edges');
    if (!edgeLayer || !this.graph) return;

    edgeLayer.innerHTML = '';
    this.graph.edges.forEach((edge: GraphEdge) => {
      const a = this.nodes.get(edge.u);
      const b = this.nodes.get(edge.v);
      if (!a || !b) return;
      const d = this.edgeShape(edge, a, b);
      edgeLayer.appendChild(el('path', { d, class: 'edge-casing' }));
      const line = el('path', { d, class: 'edge' });
      const tip = el('title', {});
      tip.textContent =
        edge.id + ' · ' + Math.round(edge.metres) + ' m' +
        (edge.stairs ? ' · steps' : '') + (edge.covered ? ' · covered' : '');
      line.appendChild(tip);
      edgeLayer.appendChild(line);
      if (edge.stairs) this.drawStairs(edgeLayer, edge, a, b);
    });
  }

  private drawNodes(): void {
    const nodeLayer = layerOf('layer-nodes');
    if (!nodeLayer || !this.graph) return;

    nodeLayer.innerHTML = '';
    this.serviceGroups = [];
    this.nodes.forEach((node) => {
      // a waypoint is a junction, not a place: the end point of a link, never drawn
      if (node.kind === 'waypoint') return;
      const p = this.project(node.lat, node.lng);
      const g = el('g', { class: 'node ' + node.kind }) as SVGGElement;
      if (node.kind === 'facility') {
        const cat = legendKeyOf(node);
        g.classList.add('cat-' + cat);
        this.serviceGroups.push({ el: g, cat });
        g.appendChild(el('circle', { cx: p.x, cy: p.y, r: SERVICE_RADIUS }));
      } else {
        // What reaches here is a place: the blocks themselves are footprints, not nodes, and
        // the 2.5D shadow face and decorative trees of the old "building" kind are gone.
        const size = PLACE_SIZE;
        // A place no walkway reaches is still on the campus, so it is drawn -- with a
        // dashed ring, because it cannot be routed to.
        if (node.off_network) {
          g.classList.add('off-network');
          g.appendChild(
            el('circle', { cx: p.x, cy: p.y, r: size * 1.5, class: 'off-network-ring' }),
          );
        }
        g.appendChild(
          el('rect', {
            x: p.x - size / 2,
            y: p.y - size / 2,
            width: size,
            height: size,
            rx: size * 0.17,
          }),
        );
        if (node.kind === 'place') {
          // A place is filed under its own category -- a block, a main location -- so the
          // legend can show or hide those destinations independently of the amenities.
          const cat = legendKeyOf(node);
          g.classList.add('cat-' + cat);
          this.serviceGroups.push({ el: g, cat });
        }
      }
      const tip = el('title', {});
      tip.textContent = node.name + (node.service ? ' · ' + node.service : '');
      g.appendChild(tip);

      g.addEventListener('click', (ev) => {
        ev.stopPropagation();
        this.opts.onPlaceClick(node);
      });
      g.addEventListener('mouseenter', () => this.opts.onTip(node.name + (node.floor ? ' · ' + node.floor : '')));
      g.addEventListener('mouseleave', () => this.opts.onTip(null));
      nodeLayer.appendChild(g);
    });

    this.applyFilters();
  }

  private edgeShape(edge: GraphEdge, a: GraphNode, b: GraphNode): string {
    const points =
      edge.path && edge.path.length >= 2
        ? edge.path.map(([lat, lng]) => this.project(lat, lng))
        : [this.project(a.lat, a.lng), this.project(b.lat, b.lng)];
    return points.map((p, i) => (i ? 'L' : 'M') + p.x.toFixed(2) + ' ' + p.y.toFixed(2)).join(' ');
  }

  /* A flight of steps is drawn as rungs instead of a plain line, because "walk 40 m
     and you will meet steps" is different advice from "walk 40 m". Spacing follows the
     same zoom compensation as every other mark, so the rungs do not merge when you
     zoom in. */
  private drawStairs(layer: SVGGElement, edge: GraphEdge, a: GraphNode, b: GraphNode): void {
    const points =
      edge.path && edge.path.length >= 2
        ? edge.path.map(([lat, lng]) => this.project(lat, lng))
        : [this.project(a.lat, a.lng), this.project(b.lat, b.lng)];
    const spacing = 2.75;
    const half = 1.7;
    for (let i = 1; i < points.length; i += 1) {
      const p0 = points[i - 1];
      const p1 = points[i];
      const dx = p1.x - p0.x;
      const dy = p1.y - p0.y;
      const len = Math.hypot(dx, dy) || 1;
      const nx = -dy / len;
      const ny = dx / len;
      for (let d = spacing / 2; d < len; d += spacing) {
        const x = p0.x + (dx * d) / len;
        const y = p0.y + (dy * d) / len;
        layer.appendChild(
          el('line', {
            x1: (x - nx * half).toFixed(1),
            y1: (y - ny * half).toFixed(1),
            x2: (x + nx * half).toFixed(1),
            y2: (y + ny * half).toFixed(1),
            class: 'edge-stairs',
          }),
        );
      }
    }
  }

  private applyFilters(): void {
    this.serviceGroups.forEach(({ el: group, cat }) => {
      group.classList.toggle('dim', this.hidden.has(cat));
    });
  }

  // -- route ---------------------------------------------------------------
  private fitTo(points: Point[]): void {
    if (!points.length) return;
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const pad = 70;
    const x = Math.min(...xs) - pad;
    const y = Math.min(...ys) - pad;
    const w = Math.max(Math.max(...xs) + pad - x, 200);
    const h = Math.max(Math.max(...ys) + pad - y, 160);
    this.view = { x, y, w, h };
    this.applyView();
  }

  // -- interaction ---------------------------------------------------------
  private bindInteractions(): void {
    const svg = this.svg;

    const onMouseDown = (ev: MouseEvent) => {
      this.drag = { x: ev.clientX, y: ev.clientY, vx: this.view.x, vy: this.view.y };
      svg.classList.add('dragging');
    };
    const onMouseUp = () => {
      this.drag = null;
      svg.classList.remove('dragging');
    };
    const onMouseMove = (ev: MouseEvent) => {
      if (!this.drag) return;
      const rect = svg.getBoundingClientRect();
      const scale = this.view.w / rect.width;
      this.view.x = this.drag.vx - (ev.clientX - this.drag.x) * scale;
      this.view.y = this.drag.vy - (ev.clientY - this.drag.y) * scale;
      this.applyView();
    };
    const onWheel = (ev: WheelEvent) => {
      ev.preventDefault();
      const rect = svg.getBoundingClientRect();
      const cx = this.view.x + (ev.clientX - rect.left) * (this.view.w / rect.width);
      const cy = this.view.y + (ev.clientY - rect.top) * (this.view.h / rect.height);
      this.zoom(ev.deltaY > 0 ? 1.12 : 0.89, cx, cy);
    };

    svg.addEventListener('mousedown', onMouseDown);
    window.addEventListener('mouseup', onMouseUp);
    window.addEventListener('mousemove', onMouseMove);
    svg.addEventListener('wheel', onWheel, { passive: false });

    this.teardown.push(() => {
      svg.removeEventListener('mousedown', onMouseDown);
      window.removeEventListener('mouseup', onMouseUp);
      window.removeEventListener('mousemove', onMouseMove);
      svg.removeEventListener('wheel', onWheel);
    });
  }
}

// -- web-mercator helpers, for the basemap ---------------------------------
function tileXOf(lng: number, z: number): number {
  return ((lng + 180) / 360) * 2 ** z;
}

function tileYOf(lat: number, z: number): number {
  const rad = (lat * Math.PI) / 180;
  return ((1 - Math.asinh(Math.tan(rad)) / Math.PI) / 2) * 2 ** z;
}

function latAtTileY(y: number, z: number): number {
  return (Math.atan(Math.sinh(Math.PI * (1 - (2 * y) / 2 ** z))) * 180) / Math.PI;
}

function lngAtTileX(x: number, z: number): number {
  return (x / 2 ** z) * 360 - 180;
}
