"use client"

import { motion } from "framer-motion"
import { Card } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Brain, ScanLine, ArrowLeft } from "lucide-react"
import Link from "next/link"

export default function BaselinePage() {
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
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  )
}