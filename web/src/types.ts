/** The role a node plays in the graph -- not how the campus map files it (see category). */
export type NodeKind = 'place' | 'facility' | 'waypoint';

export interface GraphNode {
  id: string;
  name: string;
  lat: number;
  lng: number;
  kind: NodeKind;
  /** Published by PolyU but not reachable by any mapped walkway. */
  off_network?: boolean;
  nearest_m?: number;
  /** The campus map's own category, e.g. "Amenities" or "Main Locations". */
  category?: string;
  aliases?: string[];
  names?: Record<string, string>;
  parent?: string | null;
  building?: string | null;
  service?: string | null;
  accepts?: string[];
  floor?: string | null;
}

export interface GraphEdge {
  id: string;
  u: string;
  v: string;
  metres: number;
  covered: boolean;
  stairs: boolean;
  path: [number, number][];
}

/** One drawn shape: a building ring, or a line when `line` is set. */
export interface Footprint {
  id: string;
  ring: [number, number][];
  name?: string;
  kind?: string;
  line?: boolean;
  height?: number;
  tone?: string;
  /** The letter to print on the roof: the block code the data gives, never a guess. */
  display?: string;
}

export interface Graph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  footprints: Footprint[];
}

export interface RouteSegment {
  from: string;
  to: string;
  metres: number;
  path: [number, number][];
}

export interface RoutePoint {
  id: string;
  name: string;
  lat: number;
  lng: number;
}

export interface RouteStats {
  algorithm: string;
  nodes_expanded: number;
  frontier_peak: number;
  ms: number;
}

export interface Landmark {
  id: string;
  name: string;
  /** Metres along the walk where you pass it. */
  at_m: number;
  /** How far the walked line passes from it. */
  distance_m: number;
  lat: number;
  lng: number;
}

export interface RoutePayload {
  reachable: boolean;
  metres: number;
  path: RoutePoint[];
  segments: RouteSegment[];
  stats: RouteStats | null;
  explain: unknown[];
  /** Buildings the walk passes, in order: the names on the line. */
  landmarks?: Landmark[];
}

/** One pickable way there: what the list in the reply shows, and the line to draw. */
export interface RouteOption {
  metres: number;
  legs: number;
  flights: number;
  covered: number;
  via: string;
  start: string;
  goal: string;
  path: RoutePoint[];
  segments: { path: [number, number][] }[];
  landmarks?: Landmark[];
}

export interface FacilityCard {
  name: string;
  display_name?: string;
  display_building?: string;
  display_floor?: string;
  display_hours?: string;
  building?: string;
  floor?: string;
  metres?: number;
  hours_today?: string;
  open_now?: boolean;
  lat?: number;
  lng?: number;
  [key: string]: unknown;
}

export interface ChatPayload {
  text: string;
  lead?: string;
  speech?: string;
  speech_lang?: string;
  narrated?: boolean;
  notes?: string[];
  intent?: string;
  parser?: string;
  route: RoutePayload | null;
  /** Every route worth offering, cheapest first. Empty when the reply has no route. */
  routes?: RouteOption[];
  facilities: FacilityCard[];
  error?: string;
}

export interface KnowledgeStatus {
  documents: number;
  with_opening_hours: number;
}

export interface Status {
  mode: string;
  model: string;
  api_key_present: boolean;
  /** Whether a key for the embeddings endpoint is stored. */
  embeddings_key_present?: boolean;
  /** Whether a vector-store key is stored. */
  pinecone_key_present?: boolean;
  embeddings_model?: string;
  embeddings_base_url?: string;
  nodes: number;
  edges: number;
  knowledge?: KnowledgeStatus;
  /** Which place-name resolver is live, and why it fell back if it did. */
  places?: {
    index: string;
    vectors: number;
    mode: string;
    warning?: string;
  };
}

export type Role = 'user' | 'bot';

export interface ChatMessage {
  id: number;
  role: Role;
  payload: ChatPayload | string;
}
