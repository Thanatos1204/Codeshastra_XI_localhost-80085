"use client"

import { useState, useEffect } from "react"
import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Eye, Bell, ArrowLeft, PauseCircle } from "lucide-react"
import Link from "next/link"
import { Alert, AlertDescription } from "@/components/ui/alert"

export default function LiveFeedPage() {
  type AlertType = {
    type: string;
    message: string;
  } | null;

  const [isMonitoring, setIsMonitoring] = useState(false)
  const [alert, setAlert] = useState<AlertType>(null)
  const apiUrl = 'http://localhost:5000'

  const toggleMonitoring = () => {
    setIsMonitoring(prev => {
      const newStatus = !prev;
      const video = document.getElementById("live-video") as HTMLImageElement;
      if (newStatus) {
        setAlert({ type: "info", message: "Monitoring started. AI is now detecting changes in your space." });
      } else {
        setAlert({ type: "info", message: "Monitoring paused." });
        fetch(`${apiUrl}/api/stop-stream`, { method: 'POST' })
          .then(() => {
            if (video) video.src = '';
          });
      }
      return newStatus;
    });
  };

  // Clear alerts after 5 seconds
  useEffect(() => {
    if (alert) {
      const timer = setTimeout(() => {
        setAlert(null)
      }, 5000)
      return () => clearTimeout(timer)
    }
  }, [alert])

  return (
    <div className="min-h-screen bg-black text-white p-8">
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
          className="text-center mb-8"
        >
          <h1 className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500">
            Live Feed Mode
          </h1>
          <p className="text-xl text-gray-400 max-w-2xl mx-auto">
            Monitor your space in real-time with instant notifications for any detected changes
          </p>
        </motion.div>

        {/* Alert message */}
        {alert && (
          <motion.div 
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6"
          >
            <Alert variant={alert.type === "error" ? "destructive" : "default"} 
                  className={alert.type === "info" ? "border-blue-500 bg-blue-500/10" : ""}>
              <AlertDescription>{alert.message}</AlertDescription>
            </Alert>
          </motion.div>
        )}

        {/* Live feed video display */}
        {isMonitoring && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5 }}
            className="mb-10 rounded-lg overflow-hidden border-2 border-blue-500/50 shadow-[0_0_25px_rgba(59,130,246,0.4)]"
          >
            <div className="relative bg-gray-900 aspect-video w-full">
            <img
              id="live-video"
              src={isMonitoring ? `${apiUrl}/api/live-feed` : ''}
              alt="Live video stream"
              className="w-full h-full object-contain"
            />
              <div className="absolute top-4 right-4 bg-red-500 rounded-full h-4 w-4 animate-pulse"></div>
              <div className="absolute bottom-4 left-4 bg-black/70 text-white text-sm px-3 py-1 rounded-full">
                Live
              </div>
            </div>
          </motion.div>
        )}

        <div className="grid md:grid-cols-2 gap-8">
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            <Card className="p-8 bg-gradient-to-br from-blue-500/10 to-purple-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300">
              <div className="mb-6">
                {isMonitoring ? (
                  <PauseCircle className="h-12 w-12 text-blue-500" />
                ) : (
                  <Eye className="h-12 w-12 text-blue-500" />
                )}
              </div>
              <h2 className="text-2xl font-bold mb-4">
                {isMonitoring ? "Pause Monitoring" : "Live Monitor"}
              </h2>
              <p className="text-gray-400 mb-6">
                {isMonitoring 
                  ? "Your space is currently being monitored in real-time. Click below to pause the monitoring."
                  : "Start real-time monitoring of your space. Our AI will continuously scan and compare against the baseline to detect any changes."}
              </p>
              <Button 
                onClick={toggleMonitoring}
                className={`w-full text-lg py-6 rounded-lg transition-all duration-300 ${
                  isMonitoring 
                    ? "bg-gray-600 hover:bg-gray-700 shadow-[0_0_15px_rgba(75,85,99,0.5)] hover:shadow-[0_0_30px_rgba(75,85,99,0.8)]" 
                    : "bg-blue-600 hover:bg-blue-700 shadow-[0_0_15px_rgba(59,130,246,0.5)] hover:shadow-[0_0_30px_rgba(59,130,246,0.8)]"
                }`}
              >
                {isMonitoring ? "Pause Monitoring" : "Start Monitoring"}
              </Button>
            </Card>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Card className="p-8 bg-gradient-to-br from-purple-500/10 to-pink-500/10 border-gray-800 hover:border-gray-700 transition-all duration-300">
              <div className="mb-6">
                <Bell className="h-12 w-12 text-purple-500" />
              </div>
              <h2 className="text-2xl font-bold mb-4">Notifications</h2>
              <p className="text-gray-400 mb-6">
                Configure your notification preferences and view the history of detected changes in your space.
              </p>
              <Button className="w-full bg-purple-600 hover:bg-purple-700 text-lg py-6 rounded-lg shadow-[0_0_15px_rgba(147,51,234,0.5)] hover:shadow-[0_0_30px_rgba(147,51,234,0.8)] transition-all duration-300">
                View Alerts
              </Button>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  )
}