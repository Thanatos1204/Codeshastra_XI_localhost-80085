export interface RoomScanResult {
    status: string;
    room_id: string;
    message?: string;
    timestamp: string;
    boxes: {
        id: string;
        label: string;
        bbox: [number, number, number, number];
        }[];
    original_image_url: string;
    scans: {
      original: string;
      edges: string;
      depth: string;
      depth_colored: string;
      corners: string;
      lines: string;
      all: string;
    };
    images?: {
      all: string;
      edges: string;
      depth: string;
      corners: string;
    };
  }
  
  export interface ScanDifference {
    status: string;
    room_id: string;
    comparison_id: string;
    metrics: {
      total_difference_percent: number;
      edge_difference_percent: number;
      significant_changes: boolean;
    };
    comparison_url: string;
    comparison_image?: string;
  }
  
  export interface RoomMetadata {
    room_id: string;
    timestamp: string;
    thumbnail: string;
  }