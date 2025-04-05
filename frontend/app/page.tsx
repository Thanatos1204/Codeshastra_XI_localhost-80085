"use client"

import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import Link from "next/link"
import {
  Brain,
  ScanLine,
  Eye,
  Zap,
  ArrowRight
} from "lucide-react"
import { motion } from "framer-motion"
import { BackgroundLines } from "@/components/ui/background-lines"

export default function Home() {
  const features = [
    {
      icon: <Brain className="h-8 w-8 text-blue-500" />,
      title: "Room Mapping",
      description: "Advanced AI algorithms create detailed 3D maps of any space"
    },
    {
      icon: <ScanLine className="h-8 w-8 text-purple-500" />,
      title: "Object-Level Detection",
      description: "Precise detection of changes down to individual objects"
    },
    {
      icon: <Eye className="h-8 w-8 text-pink-500" />,
      title: "Live Monitoring",
      description: "Real-time surveillance and instant change notifications"
    },
    {
      icon: <Zap className="h-8 w-8 text-yellow-500" />,
      title: "Edge Optimization",
      description: "Optimized for edge devices with minimal latency"
    }
  ]

  const timeline = [
    { title: "Scan", description: "Initial space scanning" },
    { title: "Compare", description: "AI-powered comparison" },
    { title: "Detect", description: "Change detection" },
    { title: "Monitor", description: "Continuous monitoring" }
  ]

  return (
    <div className="min-h-screen bg-black text-white">
      {/* Hero Section with BackgroundLines */}
      <BackgroundLines className="relative h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-blue-500/10 via-purple-500/10 to-pink-500/10" />
        <div className="relative z-10 text-center px-4 max-w-4xl mx-auto">
          <motion.h1
            className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            AI-Powered Space Change Detection
          </motion.h1>
          <motion.p
            className="text-xl md:text-2xl text-gray-300 mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            Scan, Compare & Monitor Spaces in Real-Time
          </motion.p>
          <motion.div
            className="flex flex-col md:flex-row gap-6 justify-center"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Button
              asChild
              className="bg-blue-600 hover:bg-blue-700 text-white font-semibold text-lg px-8 py-6 rounded-full shadow-[0_4px_20px_rgba(59,130,246,0.6)] hover:shadow-[0_8px_25px_rgba(59,130,246,0.8)] transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 relative overflow-hidden group"
            >
              <Link href="/baseline" className="flex items-center justify-center">
                <span className="relative z-10">Baseline Mode</span>
                <span className="absolute inset-0 bg-gradient-to-r from-blue-500 to-blue-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              </Link>
            </Button>

            <Button
              asChild
              className="bg-purple-600 hover:bg-purple-700 text-white font-semibold text-lg px-8 py-6 rounded-full shadow-[0_4px_20px_rgba(147,51,234,0.6)] hover:shadow-[0_8px_25px_rgba(147,51,234,0.8)] transition-all duration-300 transform hover:-translate-y-1 hover:scale-105 relative overflow-hidden group"
            >
              <Link href="/livefeed" className="flex items-center justify-center">
                <span className="relative z-10">Live Feed Mode</span>
                <span className="absolute inset-0 bg-gradient-to-r from-purple-500 to-purple-700 opacity-0 group-hover:opacity-100 transition-opacity duration-300"></span>
              </Link>
            </Button>
          </motion.div>
        </div>
      </BackgroundLines>

      {/* Features Section */}
      <section className="py-20 px-4 bg-black/50">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">Key Features</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <Card className="p-6 bg-gray-900/50 border-gray-800 hover:border-gray-700 transition-all duration-300">
                  <div className="mb-4">{feature.icon}</div>
                  <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                  <p className="text-gray-400">{feature.description}</p>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Timeline Section */}
      <section className="py-20 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-3xl md:text-4xl font-bold text-center mb-16">How It Works</h2>
          <div className="flex flex-col md:flex-row justify-between items-center md:items-start gap-8">
            {timeline.map((step, index) => (
              <motion.div
                key={step.title}
                className="flex flex-col items-center text-center"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <div className="w-16 h-16 rounded-full bg-gradient-to-r from-blue-500 to-purple-500 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(147,51,234,0.5)]">
                  <span className="text-xl font-bold">{index + 1}</span>
                </div>
                <h3 className="text-xl font-semibold mb-2">{step.title}</h3>
                <p className="text-gray-400">{step.description}</p>
                {index < timeline.length - 1 && (
                  <ArrowRight className="hidden md:block h-8 w-8 text-gray-600 absolute -right-4 top-1/2 transform -translate-y-1/2" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 px-4 border-t border-gray-800">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center mb-4 md:mb-0">
            <Brain className="h-8 w-8 text-blue-500 mr-2" />
            <span className="text-xl font-bold">SpaceWatch AI</span>
          </div>
          <nav>
            <ul className="flex gap-8">
              <li><Link href="/about" className="text-gray-400 hover:text-white transition-colors">About</Link></li>
              <li><Link href="/features" className="text-gray-400 hover:text-white transition-colors">Features</Link></li>
              <li><Link href="/contact" className="text-gray-400 hover:text-white transition-colors">Contact</Link></li>
            </ul>
          </nav>
        </div>
      </footer>
    </div>
  )
}