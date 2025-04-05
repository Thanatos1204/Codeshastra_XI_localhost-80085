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
      <TabsContent value="all" className="relative aspect-video rounded-lg overflow-hidden">
        {loading.all ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.all ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.all}`}
            alt="Combined scan result"
            objectFit="contain"
            width={1024}
            height={1024}
            className="rounded-lg w-full"
          />
        </div>
      </TabsContent>

      <TabsContent value="edges" className="relative aspect-video rounded-lg overflow-hidden">
        {loading.edges ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.edges ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.edges}`}
            alt="Edge detection"
            objectFit="contain"
            width={1024}
            height={1024}
            className="rounded-lg w-full"
          />
        </div>
      </TabsContent>

      <TabsContent value="depth" className="relative aspect-video rounded-lg overflow-hidden">
        {loading.depth ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.depth ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.depth_colored}`}
            alt="Depth map"
            objectFit="contain"
            width={1024}
            height={1024}
            className="rounded-lg w-full"
          />
        </div>
      </TabsContent>

      <TabsContent value="corners" className="relative aspect-video rounded-lg overflow-hidden">
        {loading.corners ? (
          <Skeleton className="w-full h-full absolute" />
        ) : null}
        <div className={loading.corners ? "opacity-0" : "opacity-100 transition-opacity duration-300"}>
          <Image 
            src={`http://localhost:5000${scanResult.scans.corners}`}
            alt="Corner features"
            width={1024}
            height={1024}
            objectFit="contain"
            className="rounded-lg w-full"
          />
        </div>
      </TabsContent>
    </div>
  );
}