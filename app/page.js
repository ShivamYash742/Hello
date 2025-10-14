'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { ArrowRight, Upload, Layers, Sparkles, BarChart3, Map, Play } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export default function HomePage() {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetch('/api/auth/me')
      .then(res => res.ok ? res.json() : null)
      .then(data => setUser(data))
      .catch(() => setUser(null));
  }, []);

  const pipelineStages = [
    {
      icon: Upload,
      title: 'Ingest Data',
      description: 'Upload HR optical (RGB GeoTIFF) and LR thermal IR imagery',
      color: 'from-blue-500 to-cyan-500',
      href: '/ingest'
    },
    {
      icon: Layers,
      title: 'Align',
      description: 'Multi-sensor registration with keypoint matching and optical flow',
      color: 'from-cyan-500 to-teal-500',
      href: '/align'
    },
    {
      icon: Sparkles,
      title: 'Fuse & Super-Resolve',
      description: 'Guided SR using optical edges/textures to upsample thermal 2×-4×',
      color: 'from-teal-500 to-green-500',
      href: '/fuse-sr'
    },
    {
      icon: BarChart3,
      title: 'Evaluate',
      description: 'Compute PSNR, SSIM, RMSE(K) with visual quality assessment',
      color: 'from-green-500 to-emerald-500',
      href: '/evaluate'
    },
  ];

  const useCases = [
    { title: 'Urban Planning', description: 'High-resolution thermal mapping for urban heat islands' },
    { title: 'Wildfire Monitoring', description: 'Precise temperature mapping for fire detection' },
    { title: 'Precision Agriculture', description: 'Crop stress analysis with enhanced thermal detail' },
  ];

  return (
    <div className="min-h-screen">
      <Navbar user={user} />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-[#0E88D3]/10 via-background to-[#F47216]/10" />
        <div className="container mx-auto relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-6">
            <Badge variant="outline" className="px-4 py-1">
              ISRO Thermal Super-Resolution Laboratory
            </Badge>
            <h1 className="text-5xl md:text-6xl font-bold tracking-tight">
              Optics-Guided,{' '}
              <span className="bg-gradient-to-r from-[#0E88D3] to-[#F47216] bg-clip-text text-transparent">
                Physics-Grounded
              </span>
              {' '}Thermal SR
            </h1>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Transform low-resolution thermal imagery into high-resolution temperature maps using 
              optical guidance and physics-based constraints.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-4 pt-4">
              <Button size="lg" asChild className="bg-[#0E88D3] hover:bg-[#0E88D3]/90">
                <Link href="/ingest">
                  <Upload className="mr-2 h-5 w-5" />
                  Ingest Data
                </Link>
              </Button>
              <Button size="lg" variant="outline" asChild>
                <Link href="/demo">
                  <Play className="mr-2 h-5 w-5" />
                  Run Demo
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* Pipeline Overview */}
      <section className="py-20 px-4 bg-muted/30">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Processing Pipeline</h2>
            <p className="text-muted-foreground text-lg">
              End-to-end workflow from data ingestion to evaluation
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6 max-w-7xl mx-auto">
            {pipelineStages.map((stage, index) => (
              <Card key={stage.title} className="relative overflow-hidden group hover:shadow-lg transition-all">
                <div className={`absolute top-0 left-0 right-0 h-1 bg-gradient-to-r ${stage.color}`} />
                <CardHeader>
                  <div className={`w-12 h-12 rounded-lg bg-gradient-to-r ${stage.color} flex items-center justify-center mb-4`}>
                    <stage.icon className="h-6 w-6 text-white" />
                  </div>
                  <CardTitle className="flex items-center gap-2">
                    <span className="text-muted-foreground font-mono text-sm">0{index + 1}</span>
                    {stage.title}
                  </CardTitle>
                  <CardDescription>{stage.description}</CardDescription>
                </CardHeader>
                <CardContent>
                  <Button variant="ghost" size="sm" asChild className="w-full group-hover:bg-accent">
                    <Link href={stage.href}>
                      Go to {stage.title}
                      <ArrowRight className="ml-2 h-4 w-4" />
                    </Link>
                  </Button>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Use Cases */}
      <section className="py-20 px-4">
        <div className="container mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Applications</h2>
            <p className="text-muted-foreground text-lg">
              Real-world use cases for thermal super-resolution
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            {useCases.map((useCase) => (
              <Card key={useCase.title}>
                <CardHeader>
                  <CardTitle className="text-xl">{useCase.title}</CardTitle>
                  <CardDescription>{useCase.description}</CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-[#0E88D3]/10 to-[#F47216]/10">
        <div className="container mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold mb-4">Ready to Get Started?</h2>
          <p className="text-muted-foreground text-lg mb-8 max-w-2xl mx-auto">
            Upload your optical and thermal imagery to experience the power of physics-grounded super-resolution.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-4">
            <Button size="lg" asChild className="bg-[#0E88D3] hover:bg-[#0E88D3]/90">
              <Link href="/ingest">
                <Upload className="mr-2 h-5 w-5" />
                Upload Data
              </Link>
            </Button>
            <Button size="lg" variant="outline" asChild>
              <Link href="/map">
                <Map className="mr-2 h-5 w-5" />
                View Interactive Map
              </Link>
            </Button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t py-8 px-4">
        <div className="container mx-auto text-center text-sm text-muted-foreground">
          <p>© 2025 ISRO Thermal SR Lab. Built with Next.js, Prisma, and PostgreSQL.</p>
        </div>
      </footer>
    </div>
  );
}