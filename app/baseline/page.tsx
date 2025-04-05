"use client"

import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, ScanLine, ArrowLeft } from "lucide-react"
import Link from "next/link"
import { useState, useRef, useEffect } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, ScanLine, ArrowLeft, Upload, Image as ImageIcon, RefreshCw } from "lucide-react"
import Link from "next/link"
import Image from "next/image"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { cn } from "@/lib/utils"
import { RoomScanResult } from "@/types/scan"
import ScanResultViewer from "@/components/scan-result-viewer"
import PreviousScansModal from "@/components/previous-scans-modal"

export default function BaselinePage() {
  // State for the scan process
  const [isScanning, setIsScanning] = useState(false)
  const [scanProgress, setScanProgress] = useState(0)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [scanResult, setScanResult] = useState<RoomScanResult | null>(null)
  const [activeTab, setActiveTab] = useState<string>("original")
  const [showPreviousScans, setShowPreviousScans] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  
  // Refs for UI elements
  const fileInputRef = useRef<HTMLInputElement>(null)
  const scanLineRef = useRef<HTMLDivElement>(null)

  // Handle file selection
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setSelectedImage(file)
      
      // Create preview URL
      const fileUrl = URL.createObjectURL(file)
      setPreviewUrl(fileUrl)
    }
  }

  // Handle file drop
  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setSelectedImage(file)
      
      // Create preview URL
      const fileUrl = URL.createObjectURL(file)
      setPreviewUrl(fileUrl)
    }
  }

  // Handle drag events
  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
  }

  // Trigger file input click
  const handleUploadClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click()
    }
  }

  // Start scan process
  const startScan = async () => {
    if (!selectedImage) return
    
    setIsScanning(true)
    setScanProgress(0)
    setIsUploading(true)
    
    // Create form data
    const formData = new FormData()
    formData.append('image', selectedImage)
    formData.append('include_images', 'false') // We'll fetch images via URL

    try {
      // Execute scan animation
      const scanInterval = setInterval(() => {
        setScanProgress(prev => {
          if (prev >= 100) {
            clearInterval(scanInterval)
            return 100
          }
          return prev + 2
        })
      }, 50)

      // Send API request
      const response = await fetch("http://localhost:5000/api/scan", {
        method: 'POST',
        body: formData,
      })

      clearInterval(scanInterval)
      setScanProgress(100)
      
      if (!response.ok) {
        throw new Error('Failed to process scan')
      }

      const result = await response.json()
      console.log('Scan result:', result);
      setScanResult(result)
      setActiveTab("all")
      setIsUploading(false)
      
      // Delay to show 100% progress before ending
      setTimeout(() => {
        setIsScanning(false)
      }, 1000)
    } catch (error) {
      console.error('Scan error:', error)
      setIsScanning(false)
      setIsUploading(false)
    }
  }

  // Reset state
  const resetScan = () => {
    setSelectedImage(null)
    setPreviewUrl(null)
    setScanResult(null)
    setScanProgress(0)
  }

  // Cleanup URLs when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  return (
    <div className="min-h-screen bg-black text-white md:p-8 p-4">
      {/* Header with back button */}
      <div className="max-w-7xl mx-auto mb-12">
        <Button
          asChild
          variant="ghost"
          className="text-gray-400 hover:text-white"
        >
          <Link href="/" className="flex items-center gap-2">
            <ArrowLeft className="h-4 w-4" />
            Back to Home
          </Link>
        </Button>
      </div>

      <div className="max-w-7xl mx-auto">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-16"
        >
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
            Baseline Mode
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Create a detailed baseline scan of your space for future comparison and change detection
          </p>
        </motion.div>

        <div className="grid md:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <Card className="p-8 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300">
            className="flex flex-col"
          >
            <Card className="md:p-8 p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300 h-full flex flex-col">
              <div className="mb-6">
                <Brain className="h-12 w-12 text-blue-500" />
              </div>
              <h2 className="text-2xl font-bold mb-4">Initial Scan</h2>
              <p className="text-gray-400 mb-6">
                Our AI system will perform a comprehensive scan of your space, creating a detailed 3D map and inventory of all objects and their positions.
              </p>
              <Button className="w-full bg-blue-600 hover:bg-blue-700 text-lg py-6 rounded-lg shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_30px_rgba(59,130,246,0.8)] transition-all duration-300">
                Start Scan
              </Button>
                Our AI system will perform a comprehensive scan of your space, creating a detailed map of all objects and their positions.
              </p>
              
              {/* Upload area */}
              <div className="mt-auto">
                {!selectedImage ? (
                  <div
                    className="border-2 border-dashed border-gray-700 rounded-lg p-8 mb-6 text-center cursor-pointer hover:border-blue-500 transition-colors"
                    onClick={handleUploadClick}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                  >
                    <Upload className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                    <p className="text-gray-400 mb-2">Drag and drop an image or click to upload</p>
                    <p className="text-gray-600 text-sm">Supports JPG, PNG</p>
                    <input 
                      type="file" 
                      ref={fileInputRef} 
                      onChange={handleFileChange} 
                      className="hidden" 
                      accept="image/jpeg,image/png"
                    />
                  </div>
                ) : (
                  <div className="relative mb-6 h-auto w-full rounded-lg object-contain">
                    {/* Image preview */}
                    {previewUrl && (
                      <Image 
                        src={previewUrl} 
                        alt="Room preview" 
                        objectFit="contain"
                        width={0}
                        height={0}
                        sizes="100vw"
                        className="w-auto h-auto max-w-full rounded-lg object-contain"
                      />
                    )}
                    
                    {/* Scan animation overlay */}
                    {isScanning && (
                      <div className="absolute inset-0 z-10">
                        {/* Scan line */}
                        <div 
                          ref={scanLineRef}
                          className="absolute left-0 right-0 h-1 bg-blue-500 shadow-[0_0_10px_rgba(59,130,246,0.8)]"
                          style={{ top: `${scanProgress}%` }}
                        ></div>
                        
                        {/* Progress overlay */}
                        <div className="absolute inset-0 bg-black/30 flex items-center justify-center">
                          <div className="text-center">
                            <p className="text-white font-semibold text-xl mb-2">Scanning...</p>
                            <p className="text-blue-400">{scanProgress}%</p>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    {/* Reset button */}
                    {!isScanning && (
                      <Button 
                        variant="ghost" 
                        className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 rounded-full w-8 h-8 p-0"
                        onClick={resetScan}
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                )}
                
                <Button 
                  className={cn(
                    "w-full bg-blue-600 hover:bg-blue-700 text-lg py-6 rounded-lg shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_30px_rgba(59,130,246,0.8)] transition-all duration-300",
                    (!selectedImage || isScanning) && "opacity-50 cursor-not-allowed"
                  )}
                  disabled={!selectedImage || isScanning}
                  onClick={startScan}
                >
                  {isUploading ? "Processing..." : "Start Scan"}
                </Button>
              </div>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Card className="p-8 bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300">
              <div className="mb-6">
                <ScanLine className="h-12 w-12 text-purple-500" />
              </div>
              <h2 className="text-2xl font-bold mb-4">Previous Scans</h2>
              <p className="text-gray-400 mb-6">
                View and manage your previous baseline scans. Compare different timestamps and track changes over time.
              </p>
              <Button className="w-full bg-purple-600 hover:bg-purple-700 text-lg py-6 rounded-lg shadow-[0_0_15px_rgba(147,51,234,0.5)] hover:shadow-[0_0_30px_rgba(147,51,234,0.8)] transition-all duration-300">
                View History
              </Button>
            <Card className={cn(
              "md:p-8 p-4 bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300",
              !scanResult && "h-full flex flex-col"
            )}>
              {!scanResult ? (
                <>
                  <div className="mb-6">
                    <ScanLine className="h-12 w-12 text-purple-500" />
                  </div>
                  <h2 className="text-2xl font-bold mb-4">Previous Scans</h2>
                  <p className="text-gray-400 mb-6">
                    View and manage your previous baseline scans. Compare different timestamps and track changes over time.
                  </p>
                  <Button 
                    className="w-full mt-auto bg-purple-600 hover:bg-purple-700 text-lg py-6 rounded-lg shadow-[0_0_15px_rgba(147,51,234,0.5)] hover:shadow-[0_0_30px_rgba(147,51,234,0.8)] transition-all duration-300"
                    onClick={() => setShowPreviousScans(true)}
                  >
                    View History
                  </Button>
                </>
              ) : (
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-bold">Scan Results</h2>
                    <Button 
                      variant="ghost" 
                      className="text-purple-400 hover:text-purple-300"
                      onClick={() => setShowPreviousScans(true)}
                    >
                      View History
                    </Button>
                  </div>
                  
                  <Tabs value={activeTab} onValueChange={setActiveTab} className="mb-8">
                  <TabsList className="grid  grid-cols-4 gap-2 mb-4 ">
  <TabsTrigger value="all">Combined</TabsTrigger>
  <TabsTrigger value="edges">Edges</TabsTrigger>
  <TabsTrigger value="depth">Depth</TabsTrigger>
  <TabsTrigger value="corners">Features</TabsTrigger>
</TabsList>
                    
                    <ScanResultViewer scanResult={scanResult} activeTab={activeTab} />
                  </Tabs>
                  
                  <div className="mt-4">
                    <p className="text-sm text-gray-400 mb-2">
                      Room ID: <span className="text-black">{scanResult.room_id}</span>
                    </p>
                    <p className="text-sm text-gray-400">
                      Timestamp: <span className="text-black">{new Date(scanResult.timestamp).toLocaleString()}</span>
                    </p>
                  </div>
                </div>
              )}
            </Card>
          </motion.div>
        </div>
      </div>
      
      {/* Previous scans modal */}
      <PreviousScansModal 
        isOpen={showPreviousScans} 
        onClose={() => setShowPreviousScans(false)}
        onSelect={(room) => {
          setScanResult(room)
          setActiveTab("all")
          setShowPreviousScans(false)
        }}
      />
    </div>
  )
}