'use client';

import { useState, useEffect } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import Link from 'next/link';
import { Download, BarChart3, FileText, ImageIcon, RefreshCw } from 'lucide-react';

export default function EvaluatePage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  const fetchEvaluations = async () => {
    try {
      const response = await fetch('/api/evaluations');
      const result = await response.json();
      if (result.success) {
        setData(result.data);
      }
    } catch (error) {
      console.error('Failed to fetch evaluations:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEvaluations();
  }, []);

  if (loading) {
    return (
      <DashboardLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="flex items-center justify-center h-64">
            <RefreshCw className="h-8 w-8 animate-spin" />
            <span className="ml-2">Loading evaluation data...</span>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  if (!data) {
    return (
      <DashboardLayout>
        <div className="container mx-auto px-4 py-8">
          <div className="text-center py-12">
            <h2 className="text-2xl font-bold mb-4">No Evaluation Data</h2>
            <p className="text-muted-foreground mb-4">Process some images first to see evaluation results.</p>
            <Button asChild>
              <Link href="/fuse-sr">Go to Super-Resolution</Link>
            </Button>
          </div>
        </div>
      </DashboardLayout>
    );
  }

  const { metrics, perClassMetrics, job } = data;

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
                <BarChart3 className="h-8 w-8 text-[#0E88D3]" />
                Metrics & Quality Assessment
              </h1>
              <p className="text-muted-foreground">
                Comprehensive evaluation of super-resolution results
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Download className="mr-2 h-4 w-4" />
                Export CSV
              </Button>
              <Button variant="outline" size="sm">
                <FileText className="mr-2 h-4 w-4" />
                Export JSON
              </Button>
            </div>
          </div>

          {/* Overall Metrics */}
          <div className="grid md:grid-cols-5 gap-4">
            <Card>
              <CardHeader className="pb-3">
                <CardDescription>PSNR</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-[#0E88D3]">{metrics.psnr}</div>
                <p className="text-xs text-muted-foreground mt-1">dB (higher is better)</p>
                <Badge variant="outline" className="mt-2">Excellent</Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardDescription>SSIM</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-[#0E88D3]">{metrics.ssim}</div>
                <p className="text-xs text-muted-foreground mt-1">0-1 (higher is better)</p>
                <Badge variant="outline" className="mt-2">Excellent</Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardDescription>RMSE (K)</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold text-[#F47216]">{metrics.rmse}</div>
                <p className="text-xs text-muted-foreground mt-1">Kelvin (lower is better)</p>
                <Badge variant="outline" className="mt-2">Good</Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Edge Sharpness</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{metrics.edgeSharpness}</div>
                <p className="text-xs text-muted-foreground mt-1">0-1 (higher is better)</p>
                <Badge variant="outline" className="mt-2">Good</Badge>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="pb-3">
                <CardDescription>Thermal Bias</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="text-3xl font-bold">{metrics.thermalBias}</div>
                <p className="text-xs text-muted-foreground mt-1">K (lower is better)</p>
                <Badge variant="outline" className="mt-2">Excellent</Badge>
              </CardContent>
            </Card>
          </div>

          {/* Per-Class Metrics */}
          <Card>
            <CardHeader>
              <CardTitle>Per-Class Metrics</CardTitle>
              <CardDescription>Breakdown by land cover type</CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Land Cover Class</TableHead>
                    <TableHead className="text-right">PSNR (dB)</TableHead>
                    <TableHead className="text-right">SSIM</TableHead>
                    <TableHead className="text-right">RMSE (K)</TableHead>
                    <TableHead className="text-right">Pixels</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {perClassMetrics.map((row) => (
                    <TableRow key={row.class}>
                      <TableCell className="font-medium">{row.class}</TableCell>
                      <TableCell className="text-right">{row.psnr}</TableCell>
                      <TableCell className="text-right">{row.ssim}</TableCell>
                      <TableCell className="text-right">{row.rmse}</TableCell>
                      <TableCell className="text-right">{row.pixels.toLocaleString()}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          {/* Visual Comparison */}
          <Card>
            <CardHeader>
              <CardTitle>Visual Quality Assessment</CardTitle>
              <CardDescription>Compare reference, SR output, and error maps</CardDescription>
            </CardHeader>
            <CardContent>
              <Tabs defaultValue="comparison">
                <TabsList className="grid w-full grid-cols-3">
                  <TabsTrigger value="comparison">Side-by-Side</TabsTrigger>
                  <TabsTrigger value="swipe">Swipe Comparison</TabsTrigger>
                  <TabsTrigger value="error">Error Heatmap</TabsTrigger>
                </TabsList>
                <TabsContent value="comparison" className="mt-4">
                  <div className="grid grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-center">Original Thermal</div>
                      <div className="aspect-square rounded-lg border overflow-hidden">
                        {job.thermalFile ? (
                          <img 
                            src={`/api/images/${job.thermalFile}`} 
                            alt="Original Thermal" 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-gradient-to-br from-red-500/30 to-yellow-500/30 flex items-center justify-center">
                            <ImageIcon className="h-12 w-12 text-muted-foreground" />
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-center">SR Output</div>
                      <div className="aspect-square rounded-lg border-2 border-[#F47216] overflow-hidden">
                        {job.srFile ? (
                          <img 
                            src={`/api/images/${job.srFile}`} 
                            alt="SR Output" 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-gradient-to-br from-red-500/40 to-yellow-500/40 flex items-center justify-center">
                            <ImageIcon className="h-12 w-12 text-muted-foreground" />
                          </div>
                        )}
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm font-medium text-center">Optical Reference</div>
                      <div className="aspect-square rounded-lg border overflow-hidden">
                        {job.opticalFile ? (
                          <img 
                            src={`/api/images/${job.opticalFile}`} 
                            alt="Optical Reference" 
                            className="w-full h-full object-cover"
                          />
                        ) : (
                          <div className="w-full h-full bg-gradient-to-br from-blue-500/20 to-green-500/20 flex items-center justify-center">
                            <ImageIcon className="h-12 w-12 text-muted-foreground" />
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="swipe" className="mt-4">
                  <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <p>Interactive swipe comparison</p>
                      <p className="text-sm mt-1">Reference ↔ SR Output</p>
                    </div>
                  </div>
                </TabsContent>
                <TabsContent value="error" className="mt-4">
                  <div className="aspect-video bg-gradient-to-br from-blue-500/10 via-green-500/10 to-red-500/10 rounded-lg flex items-center justify-center">
                    <div className="text-center text-muted-foreground">
                      <p>Error heatmap visualization</p>
                      <p className="text-sm mt-1">Max error: 2.8K, Mean: {metrics.rmse}K</p>
                    </div>
                  </div>
                </TabsContent>
              </Tabs>
            </CardContent>
          </Card>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button variant="outline" asChild>
              <Link href="/fuse-sr">Back to SR</Link>
            </Button>
            <div className="flex gap-2">
              <Button variant="outline" asChild>
                <Link href="/map">View on Map</Link>
              </Button>
              <Button asChild className="bg-[#0E88D3] hover:bg-[#0E88D3]/90">
                <Link href="/jobs">View All Jobs</Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}