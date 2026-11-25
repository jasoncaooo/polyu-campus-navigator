import type { GraphNode } from '../types';

export const METRES_PER_DEG_LAT = 111320;

/** Deterministic pseudo-random in [0,1) from a string, so decoration does not shimmer. */
export function hash01(text: string, salt: number): number {
  let h = 2166136261 ^ salt;
  for (let i = 0; i < text.length; i += 1) {
    h ^= text.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return ((h >>> 0) % 1000) / 1000;
}

/* ------------------------------------------------------------------- legend
 *
 * The legend is the PolyU campus map's OWN taxonomy, not a shape invented for this
 * map: every place is published under one of six categories, and a place filed under
 * "Amenities" also carries the amenity it is (a toilet, a water station, an AED).
 * So the legend sorts by category, and lists the amenity subtypes under Amenities.
 *
 * One key per chip. The key colours the pin, filters the layer, and labels the chip,
 * so an amenity that is added to the source appears in exactly one place in the code.
 */
interface LegendEntry {
  key: string;
  label: string;
}

interface LegendSection {
  id: string;
  label: string;
  entries: LegendEntry[];
}

export const LEGEND: LegendSection[] = [
  {
    id: 'places',
    label: 'Places',
    entries: [
      { key: 'buildings', label: 'blocks & buildings' },
      { key: 'locations', label: 'main locations' },
    ],
  },
  /* Public transport is open to everyone who walks onto campus, not an office, so it is
     filed with the amenities. "transport" keeps its own key because it comes from the
     published category, not from an amenity subtype. */
  {
    id: 'amenities',
    label: 'Amenities',
    entries: [
      { key: 'transport', label: 'public transport' },
      { key: 'amenity-lecture-theatre', label: 'lecture theatre' },
      { key: 'amenity-aed', label: 'AED' },
      { key: 'amenity-water', label: 'water station' },
      { key: 'amenity-toilet', label: 'toilet' },
      { key: 'amenity-accessible-toilet', label: 'accessible toilet' },
      { key: 'amenity-catering', label: 'catering outlet' },
      { key: 'amenity-clinic', label: 'clinic' },
      { key: 'amenity-lactation', label: 'lactation room' },
      { key: 'amenity-prayer', label: 'prayer room' },
      { key: 'amenity-other', label: 'other amenity' },
    ],
  },
  {
    id: 'offices',
    label: 'Offices',
    entries: [
      { key: 'faculties', label: 'faculties & schools' },
      { key: 'offices', label: 'administrative offices' },
    ],
  },
];

/** Every chip, in the order the legend draws them. */
export const ALL_LEGEND_KEYS: string[] = LEGEND.flatMap((section) =>
  section.entries.map((entry) => entry.key),
);

const CATEGORY_KEY: Record<string, string> = {
  'buildings / cores / wings / blocks': 'buildings',
  'main locations': 'locations',
  'faculties, schools & departments': 'faculties',
  'administrative & supporting office': 'offices',
  'public transport': 'transport',
};

/** The amenity subtype, from the service label the engine sends. */
function amenityKey(service: string): string {
  const s = service.toLowerCase();
  if (s.includes('accessible') && s.includes('toilet')) return 'amenity-accessible-toilet';
  if (s.includes('toilet')) return 'amenity-toilet';
  if (s.includes('water')) return 'amenity-water';
  if (s.includes('aed') || s.includes('defibrillator')) return 'amenity-aed';
  if (s.includes('catering') || s.includes('canteen') || s.includes('cafe')) return 'amenity-catering';
  if (s.includes('lecture')) return 'amenity-lecture-theatre';
  if (s.includes('clinic')) return 'amenity-clinic';
  if (s.includes('lactation') || s.includes('nursing')) return 'amenity-lactation';
  if (s.includes('prayer')) return 'amenity-prayer';
  return 'amenity-other';
}

/**
 * Which chip a node belongs to.
 *
 * The published category decides, and only "Amenities" is split further by the service
 * label. A graph built before the category was stored falls back to the service alone,
 * so an older campus_graph.json still colours and filters sensibly.
 */
export function legendKeyOf(node: GraphNode): string {
  const category = (node.category || '').toLowerCase().trim();
  const direct = CATEGORY_KEY[category];
  if (direct) return direct;
  if (category.startsWith('amenities')) return amenityKey(node.service || '');
  if (node.kind === 'place') return 'locations';
  return amenityKey(node.service || '');
}
