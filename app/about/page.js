'use client';

import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Sparkles, Zap, Shield, Target } from 'lucide-react';

export default function AboutPage() {
  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div className="text-center space-y-4">
            <div className="flex items-center justify-center">
              <div className="w-20 h-20 rounded-lg bg-gradient-to-br from-[#0E88D3] to-[#F47216] flex items-center justify-center">
                <span className="text-white font-bold text-3xl">IS</span>
              </div>
            </div>
            <h1 className="text-4xl font-bold">ISRO Thermal SR Lab</h1>
            <p className="text-xl text-muted-foreground">
              Optics-Guided, Physics-Grounded Thermal Super-Resolution
            </p>
            <div className="flex items-center justify-center gap-2">
              <Badge>Version 1.0.0</Badge>
              <Badge variant="outline">Production Ready</Badge>
            </div>
          </div>

          {/* Problem Statement */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-[#0E88D3]" />
                Problem Statement
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p className="text-muted-foreground">
                Thermal infrared sensors typically provide lower spatial resolution compared to 
                optical sensors due to physical constraints (longer wavelengths, detector technology). 
                This limits their effectiveness in applications requiring fine-grained temperature mapping.
              </p>
              <p className="text-muted-foreground">
                Our solution leverages high-resolution optical imagery as guidance to enhance thermal 
                resolution while maintaining thermodynamic consistency through physics-based constraints.
              </p>
            </CardContent>
          </Card>

          {/* Approach */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-[#F47216]" />
                Technical Approach
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-3">
                <div>
                  <h3 className="font-semibold mb-2">1. Multi-Sensor Alignment</h3>
                  <p className="text-sm text-muted-foreground">
                    Feature-based registration using ORB/AKAZE keypoint matching with RANSAC homography 
                    estimation, plus dense optical flow for local parallax correction.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">2. Optics-Guided Fusion</h3>
                  <p className="text-sm text-muted-foreground">
                    Extract edges and textures from high-resolution optical imagery to guide the 
                    super-resolution of aligned thermal data. Three model options: CNN, Guidance-Disentanglement, 
                    and Swin Transformer.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">3. Physics-Based Constraints</h3>
                  <p className="text-sm text-muted-foreground">
                    Apply radiative transfer corrections, emissivity handling, and energy-balance 
                    regularization to prevent optical texture leakage into temperature channels.
                  </p>
                </div>
                <div>
                  <h3 className="font-semibold mb-2">4. Comprehensive Evaluation</h3>
                  <p className="text-sm text-muted-foreground">
                    Compute PSNR, SSIM, and RMSE (Kelvin) metrics with per-class analysis and visual 
                    quality assessment tools.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Features */}
          <div className="grid md:grid-cols-3 gap-4">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Zap className="h-4 w-4 text-yellow-500" />
                  Real-Time Processing
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Efficient CNN models enable real-time SR for mid-sized scenes (512×512 → 2048×2048)
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Shield className="h-4 w-4 text-green-500" />
                  Physics Grounded
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  Thermal fidelity guardrails prevent unphysical temperature artifacts
                </p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2 text-base">
                  <Sparkles className="h-4 w-4 text-[#F47216]" />
                  Production Ready
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground">
                  PostgreSQL backend, auth, job queue, and REST API for integration
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Use Cases */}
          <Card>
            <CardHeader>
              <CardTitle>Applications</CardTitle>
              <CardDescription>Real-world use cases</CardDescription>
            </CardHeader>
            <CardContent>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li className="flex items-start gap-2">
                  <span className="text-[#0E88D3] font-bold mt-1">•</span>
                  <span><strong>Urban Planning:</strong> High-resolution thermal mapping for urban heat island analysis</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#0E88D3] font-bold mt-1">•</span>
                  <span><strong>Wildfire Monitoring:</strong> Precise temperature mapping for early fire detection</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#0E88D3] font-bold mt-1">•</span>
                  <span><strong>Precision Agriculture:</strong> Crop stress analysis with enhanced thermal detail</span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-[#0E88D3] font-bold mt-1">•</span>
                  <span><strong>Industrial Inspection:</strong> Equipment temperature monitoring at fine scales</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          {/* Credits */}
          <Card>
            <CardHeader>
              <CardTitle>Credits & Acknowledgments</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm text-muted-foreground">
              <p>
                This prototype was developed as part of ISRO's thermal imaging research program, 
                leveraging state-of-the-art deep learning techniques and geospatial processing.
              </p>
              <p className="text-xs">
                Technologies: Next.js 14, React 18, Prisma, PostgreSQL, TailwindCSS, shadcn/ui
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </DashboardLayout>
  );
}