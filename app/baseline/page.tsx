"use client"

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
  const [nextSelectedImage, setNextSelectedImage] = useState<File | null>(null)
  const [nextPreviewUrl, setNextPreviewUrl] = useState<string | null>(null)
  const [isScanningNext, setNextIsScanning] = useState(false)
  const [nextScanResult, setNextScanResult] = useState<RoomScanResult | null>(null)
  
  const [activeTab, setActiveTab] = useState<string>("original")
  const [showPreviousScans, setShowPreviousScans] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [nextScan, setNextScan] = useState(false)
  const [comparisonResult, setComparisonResult] = useState<any>(null)
  const [comparisonError, setComparisonError] = useState<string | null>(null)
  // New state variables
  const [showDetailsModal, setShowDetailsModal] = useState(false);
  const [detailsData, setDetailsData] = useState<{
    image_base64: string;
    stats: {
      total_in_base: number;
      total_in_current: number;
      total_difference: number;
      total_added: number;
      total_removed: number;
      total_moved: number;
      increase_in_objects: number;
      decrease_in_objects: number;
    } | null;
  } | null>(null);
  const [isLoadingDetails, setIsLoadingDetails] = useState(false);

  // Function to fetch detailed comparison
  const fetchDetailedComparison = async () => {
    if (!selectedImage || !nextSelectedImage) return;
    
    setIsLoadingDetails(true);
    
    try {
      const formData = new FormData();
      formData.append('base', selectedImage);
      formData.append('current', nextSelectedImage);
      
      const response = await fetch("http://localhost:5001/compare", {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch detailed comparison');
      }
      
      const data = await response.json();
      setDetailsData(data);
      setShowDetailsModal(true);
      setNextScan(false);
    } catch (error) {
      console.error('Detailed comparison error:', error);
    } finally {
      setIsLoadingDetails(false);
    }
  };
  
  // Refs for UI elements
  const fileInputRef = useRef<HTMLInputElement>(null)
  const nextFileInputRef = useRef<HTMLInputElement>(null)
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

  const handleNextFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0]
      setNextSelectedImage(file)
      
      // Create preview URL
      const fileUrl = URL.createObjectURL(file)
      setNextPreviewUrl(fileUrl)
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

  const handleNextDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0]
      setNextSelectedImage(file)
      
      // Create preview URL
      const fileUrl = URL.createObjectURL(file)
      setNextPreviewUrl(fileUrl)
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

  const handleNextUploadClick = () => {
    if (nextFileInputRef.current) {
      nextFileInputRef.current.click()
    }
  }


  const proceedToNextScan = () => {
    setNextScan(true)
    setNextScanResult(null)
    setNextSelectedImage(null)
    setNextPreviewUrl(null)
    setNextIsScanning(false)
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
  // // Reset state
  // const resetScan = () => {
  //   setSelectedImage(null)
  //   setPreviewUrl(null)
  //   setScanResult(null)
  //   setScanProgress(0)
 }
  
  // Compare with baseline
  const handleCompareWithBaseline = async () => {
    if (!nextSelectedImage || !scanResult) return
  
    setIsUploading(true)
    setComparisonError(null)
  
    const formData = new FormData()
    formData.append('image', nextSelectedImage)
    formData.append('room_id', scanResult.room_id)
  
    try {
      const response = await fetch("http://localhost:5000/api/compare", {
        method: 'POST',
        body: formData,
      })
  
      if (!response.ok) {
        throw new Error('Failed to process comparison')
      }
  
      const result = await response.json()
      setComparisonResult(result)
    } catch (error) {
      console.error('Comparison error:', error)
      setComparisonError(error instanceof Error ? error.message : 'An unknown error occurred')
    } finally {
      setIsUploading(false)
    }
  }
  

  // Cleanup URLs when component unmounts
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl)
      }
    }
  }, [previewUrl])

  function resetScan(event: React.MouseEvent<HTMLButtonElement, MouseEvent>): void {
    // Stop propagation to prevent any parent elements from handling this click
    event.stopPropagation();
    
    // Reset all scan-related state variables
    setSelectedImage(null);
    setPreviewUrl(null);
    setScanResult(null);
    setScanProgress(0);
    setActiveTab("original");
  }

  function nextResetScan(event: React.MouseEvent<HTMLButtonElement, MouseEvent>): void {
    // Stop propagation to prevent any parent elements from handling this click
    event.stopPropagation();

    // Reset all scan-related state variables
    setNextSelectedImage(null);
    setNextPreviewUrl(null);
    setNextScanResult(null);
    setNextIsScanning(false);
    setNextScan(false);
    setComparisonResult(null);
    setComparisonError(null);
  }

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
            className="flex flex-col"
          >
            <Card className="md:p-8 p-4 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300 h-full flex flex-col">
              <div className="mb-6">
                <Brain className="h-12 w-12 text-blue-500" />
              </div>
              <h2 className="text-2xl font-bold mb-4">Initial Scan</h2>
              <p className="text-gray-400 mb-6">
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
                  <Button 
                  className={cn(
                    "w-full bg-blue-600 hover:bg-blue-700 text-lg py-6 mt-6 rounded-lg shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_30px_rgba(59,130,246,0.8)] transition-all duration-300",
                    (!selectedImage || isScanning) && "opacity-50 cursor-not-allowed"
                  )}
                  disabled={!selectedImage || isScanning}
                  onClick={proceedToNextScan}
                >
                  {isUploading ? "Processing..." : "Take Next Scan"}
                </Button>
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

      {/* Next scan comparison modal */}
      {nextScan && (
      <dialog
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
      >
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-gradient-to-br from-blue-900/90 to-purple-900/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 w-full max-w-2xl overflow-hidden"
        >
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-2xl font-bold text-white">Compare with Baseline</h2>
            <Button 
              variant="ghost"
              className="h-8 w-8 p-0 rounded-full"
              onClick={() => setNextScan(false)}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
            </Button>
          </div>
          
          {!comparisonResult ? (
            <>
              <p className="text-gray-300 mb-6">
                Upload a new image to compare with your baseline scan and detect any changes.
              </p>
              
              <div className="space-y-6">
                {/* Image upload area */}
                <div
                  className="border-2 border-dashed border-gray-700 rounded-lg p-8 text-center cursor-pointer hover:border-blue-500 transition-colors"
                  onClick={handleNextUploadClick}
                  onDrop={handleNextDrop}
                  onDragOver={handleDragOver}
                >
                  {!nextSelectedImage ? (
                    <>
                      <Upload className="h-12 w-12 text-gray-500 mx-auto mb-4" />
                      <p className="text-gray-400 mb-2">Drag and drop an image or click to upload</p>
                      <p className="text-gray-600 text-sm">Supports JPG, PNG</p>
                    </>
                  ) : (
                    <div className="relative">
                      {nextPreviewUrl && (
                        <Image
                          src={nextPreviewUrl}
                          alt="Next scan preview"
                          width={0}
                          height={0}
                          sizes="100vw"
                          className="w-auto h-auto max-w-full rounded-lg object-contain"
                        />
                      )}
                      <Button 
                        variant="ghost" 
                        className="absolute top-2 right-2 bg-black/50 hover:bg-black/70 rounded-full w-8 h-8 p-0"
                        onClick={nextResetScan}
                      >
                        <RefreshCw className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                  <input 
                    type="file"
                    ref={nextFileInputRef}
                    onChange={handleNextFileChange}
                    className="hidden"
                    accept="image/jpeg,image/png"
                  />
                </div>
                
                {/* Room ID info */}
                {comparisonResult && (
                  <div className="bg-black/30 rounded-lg p-4">
                    <p className="text-sm text-gray-400">
                      Comparing with Baseline Room: 
                      <span className="text-white ml-2">{scanResult?.room_id}</span>
                    </p>
                  </div>
                )}

                {comparisonError && (
                  <div className="bg-red-900/30 border border-red-800 rounded-lg p-4 text-red-300">
                    <p>{comparisonError}</p>
                  </div>
                )}

                {/* Buttons */}
                <div className="flex gap-4 justify-end">
                  <Button 
                    variant="outline" 
                    onClick={() => setNextScan(false)}
                    className="border-gray-600 hover:bg-gray-800"
                  >
                    Cancel
                  </Button>
                  
                  <Button
                    className={cn(
                      "bg-blue-600 hover:bg-blue-700 shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_30px_rgba(59,130,246,0.8)] transition-all duration-300",
                      (!nextSelectedImage || isUploading) && "opacity-50 cursor-not-allowed"
                    )}
                    disabled={!nextSelectedImage || isUploading}
                    onClick={handleCompareWithBaseline}
                  >
                    {isUploading ? "Processing..." : "Compare Now"}
                  </Button>
                </div>
              </div>
            </>
          ) : (
            <div className="space-y-6">
              <div className="bg-black/30 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-3 text-white">Comparison Results</h3>
                <p className="text-sm text-gray-400 mb-2">
                  Room ID: <span className="text-white">{comparisonResult.room_id}</span>
                </p>
                <p className="text-sm text-gray-400">
                  Comparison ID: <span className="text-white">{comparisonResult.comparison_id}</span>
                </p>
              </div>
              
              <div className="bg-black/30 rounded-lg p-4">
                <h3 className="text-lg font-medium mb-3 text-white">Difference Metrics</h3>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(comparisonResult.metrics).map(([key, value]) => (
                    <div key={key} className="bg-gray-900/50 rounded p-3">
                      <p className="text-sm text-gray-400">{key}:</p>
                      <p className="text-lg font-medium text-white">{typeof value === 'number' ? `${value.toFixed(2)}%` : String(value)}</p>
                    </div>
                  ))}
                </div>
              </div>
              {comparisonResult && (
                <div className="rounded-lg overflow-hidden">
                  <Image 
                    src={`http://localhost:5000${comparisonResult.comparison_url}`}
                    alt="Comparison visualization"
                    width={0}
                    height={0}
                    sizes="100vw"
                    className="w-full h-auto"
                  />
                </div>
              )}

              {/* Replace the Close Comparison button with buttons */}
             
                <div className="flex justify-end gap-3">
                  <Button 
                    className="bg-blue-600 hover:bg-blue-700"
                    onClick={fetchDetailedComparison}
                    disabled={isLoadingDetails}
                  >
                    {isLoadingDetails ? "Loading..." : "Next"}
                  </Button>
                  <Button 
                    className="bg-purple-600 hover:bg-purple-700"
                    onClick={() => {
                      setComparisonResult(null);
                      setNextScan(false);
                    }}
                  >
                    Close Comparison
                  </Button>
                </div>
            </div>
          )}
        </motion.div>
      </dialog>
        )}
              

         
              {showDetailsModal && detailsData && (
                <dialog className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
                  <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className="bg-gradient-to-br from-blue-900/90 to-purple-900/90 backdrop-blur-sm border border-gray-700 rounded-xl p-6 w-full max-w-2xl overflow-hidden"
                  >
                    <div className="flex justify-between items-center mb-4">
                      <h2 className="text-2xl font-bold text-white">Detailed Analysis</h2>
                      <Button 
                        variant="ghost"
                        className="h-8 w-8 p-0 rounded-full"
                        onClick={() => setShowDetailsModal(false)}
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
                      </Button>
                    </div>
                    
                    <div className="space-y-6">
                      {/* Visualization */}
                      <div className="rounded-lg overflow-hidden mb-4">
                        <Image 
                          src={`${detailsData.image_base64}`}
                          alt="Detailed comparison"
                          width={0}
                          height={0}
                          sizes="100vw"
                          className="w-full h-auto"
                          unoptimized={true}
                        />
                      </div>
                      
                      {/* Stats grid */}
                      <div className="bg-black/30 rounded-lg p-4">
                        <h3 className="text-lg font-medium mb-3 text-white">Object Difference Analysis</h3>
                        <div className="grid grid-cols-2 gap-4">
                          {detailsData.stats && Object.entries(detailsData.stats).map(([key, value]) => (
                            <div key={key} className="bg-gray-900/50 rounded p-3">
                              <p className="text-sm text-gray-400">{key.replace(/_/g, ' ')}:</p>
                              <p className="text-lg font-medium text-white">{value}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex justify-end">
                        <Button 
                          className="bg-purple-600 hover:bg-purple-700"
                          onClick={() => setShowDetailsModal(false)}
                        >
                          Close
                        </Button>
                      </div>
                    </div>
                  </motion.div>
                </dialog>
              )}

              
      
    

    </div>
  )
}
