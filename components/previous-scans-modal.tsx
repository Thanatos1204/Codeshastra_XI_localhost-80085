import { useState, useEffect } from "react";
import Image from "next/image";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { RoomMetadata, RoomScanResult } from "@/types/scan";

interface PreviousScansModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (room: RoomScanResult) => void;
}

export default function PreviousScansModal({
  isOpen,
  onClose,
  onSelect,
}: PreviousScansModalProps) {
  const [rooms, setRooms] = useState<RoomMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch all rooms when modal opens
  useEffect(() => {
    if (isOpen) {
      fetchRooms();
    }
  }, [isOpen]);

  const fetchRooms = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("http://localhost:5000/api/rooms");
      console.log("Response:", response);
      if (!response.ok) {
        throw new Error("Failed to fetch rooms");
      }

      const data = await response.json();
      setRooms(data.rooms || []);
    } catch (error) {
      console.error("Error fetching rooms:", error);
      setError("Failed to load previous scans. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const handleRoomSelect = async (roomId: string) => {
    try {
      const response = await fetch(`http:localhost:5000/api/rooms/${roomId}`);
      if (!response.ok) {
        throw new Error("Failed to fetch room details");
      }

      const data = await response.json();
      onSelect(data);
    } catch (error) {
      console.error("Error selecting room:", error);
    }
  };

  // Format date for display
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "numeric",
    }).format(date);
  };

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-gray-900 border-gray-800 text-white sm:max-w-3xl">
        <DialogHeader>
          <DialogTitle className="text-2xl font-bold">Previous Scans</DialogTitle>
        </DialogHeader>

        {loading ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4">
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="flex flex-col">
                <Skeleton className="w-full aspect-video rounded-lg mb-2" />
                <Skeleton className="w-3/4 h-4 mb-1" />
                <Skeleton className="w-1/2 h-3" />
              </div>
            ))}
          </div>
        ) : error ? (
          <div className="text-center py-8">
            <p className="text-red-400 mb-4">{error}</p>
            <Button
              variant="outline"
              onClick={fetchRooms}
              className="border-gray-700 hover:bg-gray-800"
            >
              Retry
            </Button>
          </div>
        ) : rooms.length === 0 ? (
          <div className="text-center py-8">
            <p className="text-gray-400">No previous scans found.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mt-4 max-h-[500px] overflow-y-auto pr-2">
            {rooms.map((room) => (
              <div
                key={room.room_id}
                className="flex flex-col border border-gray-800 rounded-lg overflow-hidden hover:border-blue-500 cursor-pointer transition-all duration-300"
                onClick={() => handleRoomSelect(room.room_id)}
              >
                <div className="relative aspect-video">
                  <Image
                    src={`http://localhost:5000${room.thumbnail}`}
                    alt={`Room ${room.room_id}`}
                    layout="fill"
                    objectFit="cover"
                  />
                </div>
                <div className="p-3">
                  <p className="font-medium truncate">Room {room.room_id.substring(0, 8)}...</p>
                  <p className="text-sm text-gray-400">
                    {formatDate(room.timestamp)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}