// ExecutionTrace Type Definitions

interface ExecutionTrace {
  envelope: EventEnvelope;
  semantic_payload: SemanticPayload;
  parser_fingerprint: ParserFingerprint;
  side_effects: SideEffect[];
  snapshot_contract?: SnapshotContract;
  branch_lineage?: BranchLineage;
}

interface EventEnvelope {
  event_id: string;
  execution_id: string;
  event_type: EventType;
  timestamp_hlc: string;
  causality: Causality;
  actor: Actor;
}

type EventType = "FINANCIAL_INTENT_CAPTURED" | "LINGUISTIC_PARSE_COMPLETED" | "SEMANTIC_TRANSITION_APPLIED";

interface Causality {
  parent_event_id?: string;
  branch_id: string;
  causality_depth?: number;
}

interface Actor {
  type: "human" | "system" | "agent";
  device_id?: string;
  user_id?: string;
}

interface SemanticPayload {
  raw_input: string;
  normalized?: { numeric_value?: number; currency?: string; confidence: number };
  linguistic_trace: LinguisticTrace;
}

interface LinguisticTrace {
  language: string;
  confidence: number;
  pipeline_version: string;
  modules: Array<{ module: string; result: unknown; execution_time_ms?: number }>;
  tokenization?: { syllable_ids?: number[]; morpheme_ids?: number[] };
}

interface ParserFingerprint {
  module: string;
  version: string;
  sha256: string;
  config_hash?: string;
}

interface SideEffect {
  effect_id: string;
  effect_type: string;
  target_system: string;
  status: "pending" | "applied" | "failed" | "reverted";
  receipt?: { applied_at: string; external_id: string; checksum: string };
}

interface SnapshotContract {
  snapshot_id: string;
  state_hash: string;
  checkpoint_event_id: string;
  recovery_point: boolean;
}

interface BranchLineage {
  branch_id: string;
  parent_branch_id?: string;
  merge_target?: string;
  conflict_resolution?: { strategy: string; resolved_at: string };
}
