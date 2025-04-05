import Image from "next/image";
import { useState, useEffect } from "react";
import { TabsContent } from "@/components/ui/tabs";
import { Skeleton } from "@/components/ui/skeleton";
import { RoomScanResult } from "@/types/scan";

interface ScanResultViewerProps {
  scanResult: RoomScanResult;
  activeTab: string;
}

export default function ScanResultViewer({ scanResult, activeTab }: ScanResultViewerProps) {
  const [loading, setLoading] = useState<Record<string, boolean>>({
    all: true,
    edges: true,
    depth: true,
    corners: true,
  });

  // Prefetch all images
  useEffect(() => {
    const prefetchImage = async (url: string, key: string) => {
      try {
        // Use Image constructor to preload
        // Make a proper API request to ensure the image is retrievable
        // Join the base URL with the image URL
        const fullUrl = `http://localhost:5000${url}`;
        const response = await fetch(fullUrl);
        const img = new window.Image();
        img.src = fullUrl;
        img.onload = () => {
          setLoading(prev => ({ ...prev, [key]: false }));
        };
      } catch (error) {
        console.error(`Failed to load image for ${key}:`, error);
      }
    };

    if (scanResult) {
      prefetchImage(scanResult.scans.all, 'all');
      prefetchImage(scanResult.scans.edges, 'edges');
      prefetchImage(scanResult.scans.depth_colored, 'depth');
      prefetchImage(scanResult.scans.corners, 'corners');
    }
  }, [scanResult]);

  if (!scanResult) return null;

  return (
    <div className="relative">
      <TabsContent value="all" className="relative mb-6 h-auto w-full rounded-lg object-contain">
        {loading.all ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.all ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.all}`}
            alt="Combined scan result"
            width={0}
            height={0}
            sizes="100vw"
            className="w-auto h-auto max-w-full rounded-lg object-contain"
          />
        </div>
      </TabsContent>

      <TabsContent value="edges" className="relative mb-6 h-auto w-full rounded-lg object-contain">
        {loading.edges ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.edges ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.edges}`}
            alt="Edge detection"
            width={0}
            height={0}
            sizes="100vw"
            className="w-auto h-auto max-w-full rounded-lg object-contain"
          />
        </div>
      </TabsContent>

      <TabsContent value="depth" className="relative mb-6 h-auto w-full rounded-lg object-contain">
        {loading.depth ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.depth ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.depth_colored}`}
            alt="Depth map"
            width={0}
            height={0}
            sizes="100vw"
            className="w-auto h-auto max-w-full rounded-lg object-contain"
          />
        </div>
      </TabsContent>

      <TabsContent value="corners" className="relative mb-6 h-auto w-full rounded-lg object-contain">
        {loading.corners ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.corners ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.corners}`}
            alt="Corner features"
            width={0}
            height={0}
            sizes="100vw"
            className="w-auto h-auto max-w-full rounded-lg object-contain"
          />
        </div>
      </TabsContent>
    </div>
  );
}