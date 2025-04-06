// components/CriticalObjectSelectorModal.tsx
"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { motion } from "framer-motion";

interface ObjectBox {
  id: string;
  label: string;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
}

interface CriticalObject {
  id: string;
  tag: string;
  isCritical: boolean;
  bbox: [number, number, number, number];
}

interface Props {
  isOpen: boolean;
  onClose: () => void;
  imageUrl: string;
  boxes: ObjectBox[];
  roomId: string;
  onSaveComplete?: () => void;
}

export default function CriticalObjectSelectorModal({
  isOpen,
  onClose,
  imageUrl,
  boxes,
  roomId,
  onSaveComplete
}: Props) {
  const [selectedBox, setSelectedBox] = useState<ObjectBox | null>(null);
  const [tag, setTag] = useState("");
  const [isCritical, setIsCritical] = useState(true);
  const [criticalObjects, setCriticalObjects] = useState<CriticalObject[]>([]);
  const imageContainerRef = useRef<HTMLDivElement>(null);
  const [imageDims, setImageDims] = useState({ width: 1, height: 1 });

  useEffect(() => {
    const updateSize = () => {
      if (imageContainerRef.current) {
        const rect = imageContainerRef.current.getBoundingClientRect();
        setImageDims({ width: rect.width, height: rect.height });
      }
    };

    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, [isOpen]);

  const handleBoxClick = (box: ObjectBox) => {
    setSelectedBox(box);
    setTag(box.label);
    setIsCritical(true);
  };

  const handleSaveTag = () => {
    if (!selectedBox || !tag) return;
    const newCritical: CriticalObject = {
      id: selectedBox.id,
      tag,
      isCritical,
      bbox: selectedBox.bbox
    };
    setCriticalObjects((prev) => [...prev.filter((o) => o.id !== selectedBox.id), newCritical]);
    setSelectedBox(null);
    setTag("");
    setIsCritical(true);
  };

  const handleSubmit = async () => {
    const response = await fetch("http://localhost:5001/save_critical", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room_id: roomId, objects: criticalObjects })
    });

    if (response.ok) {
      alert("Critical objects saved.");
      if (onSaveComplete) onSaveComplete();
      onClose();
    } else {
      alert("Error saving critical objects.");
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-5xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">Select & Tag Critical Objects</DialogTitle>
        </DialogHeader>

        <div ref={imageContainerRef} className="relative w-full">
          <Image
            src={imageUrl}
            alt="Room scan"
            width={0}
            height={0}
            sizes="100vw"
            className="w-full h-auto rounded-md"
            unoptimized
            onLoadingComplete={(img) => {
              setImageDims({ width: img.naturalWidth, height: img.naturalHeight });
            }}
          />
          {boxes.map((box) => {
            const [x1, y1, x2, y2] = box.bbox;
            const left = (x1 / imageDims.width) * 100;
            const top = (y1 / imageDims.height) * 100;
            const width = ((x2 - x1) / imageDims.width) * 100;
            const height = ((y2 - y1) / imageDims.height) * 100;

            return (
              <div
                key={box.id}
                onClick={() => handleBoxClick(box)}
                className="absolute border-2 border-blue-500 rounded-md cursor-pointer z-10"
                style={{
                  left: `${left}%`,
                  top: `${top}%`,
                  width: `${width}%`,
                  height: `${height}%`
                }}
              />
            );
          })}
        </div>

        {selectedBox && (
          <motion.div
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 50 }}
            className="bg-black/30 border border-gray-700 rounded-lg p-4 mt-4"
          >
            <h3 className="text-lg font-semibold mb-2 text-white">Tag Object</h3>
            <Input
              type="text"
              placeholder="Enter object name (e.g. Fire Extinguisher)"
              value={tag}
              onChange={(e) => setTag(e.target.value)}
              className="mb-3 text-black"
            />
            <div className="flex items-center space-x-2 mb-3">
              <Checkbox
                id="critical"
                checked={isCritical}
                onCheckedChange={() => setIsCritical(!isCritical)}
              />
              <label htmlFor="critical" className="text-white text-sm">
                Mark as critical
              </label>
            </div>
            <div className="flex justify-end gap-2">
              <Button variant="ghost" onClick={() => setSelectedBox(null)}>
                Cancel
              </Button>
              <Button onClick={handleSaveTag}>Save</Button>
            </div>
          </motion.div>
        )}

        {criticalObjects.length > 0 && (
          <div className="mt-6 text-right">
            <Button onClick={handleSubmit}>Save All Critical Tags</Button>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
