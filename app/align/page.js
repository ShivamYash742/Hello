'use client';

import { useState } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { toast } from 'sonner';
import Link from 'next/link';
import { ArrowRight, Play, RefreshCw, Crosshair } from 'lucide-react';

export default function AlignPage() {
  const [method, setMethod] = useState('orb');
  const [matchThreshold, setMatchThreshold] = useState([0.75]);
  const [ransacThreshold, setRansacThreshold] = useState([3.0]);
  const [processing, setProcessing] = useState(false);
  const [aligned, setAligned] = useState(false);
  const [metrics, setMetrics] = useState(null);

  const handleAlign = async () => {
    setProcessing(true);
    
    // Simulate processing
    setTimeout(() => {
      setMetrics({
        keypointsOptical: 1247,
        keypointsThermal: 843,
        matches: 326,
        inliers: 298,
        rmse: 1.23,
        transformMatrix: [
          [0.998, -0.062, 12.4],
          [0.062, 0.998, -8.7],
          [0, 0, 1]
        ]
      });
      setAligned(true);
      setProcessing(false);
      toast.success('Alignment completed successfully!');
    }, 2000);
  };

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2">Multi-Sensor Alignment</h1>
            <p className="text-muted-foreground">
              Automatic feature-based registration to co-register HR optical with LR thermal
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Alignment Parameters</CardTitle>
                  <CardDescription>Configure feature detection and matching</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label>Detection Method</Label>
                    <Select value={method} onValueChange={setMethod}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="orb">ORB (Oriented FAST)</SelectItem>
                        <SelectItem value="akaze">AKAZE</SelectItem>
                        <SelectItem value="sift">SIFT</SelectItem>
                        <SelectItem value="surf">SURF</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Match Threshold</Label>
                      <span className="text-sm text-muted-foreground">{matchThreshold[0].toFixed(2)}</span>
                    </div>
                    <Slider
                      value={matchThreshold}
                      onValueChange={setMatchThreshold}
                      min={0.5}
                      max={0.95}
                      step={0.05}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>RANSAC Threshold (px)</Label>
                      <span className="text-sm text-muted-foreground">{ransacThreshold[0].toFixed(1)}</span>
                    </div>
                    <Slider
                      value={ransacThreshold}
                      onValueChange={setRansacThreshold}
                      min={1.0}
                      max={10.0}
                      step={0.5}
                      className="w-full"
                    />
                  </div>

                  <Button
                    onClick={handleAlign}
                    disabled={processing}
                    className="w-full bg-[#0E88D3] hover:bg-[#0E88D3]/90"
                  >
                    {processing ? (
                      <>
                        <RefreshCw className="mr-2 h-4 w-4 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Run Alignment
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Metrics */}
              {metrics && (
                <Card>
                  <CardHeader>
                    <CardTitle>Alignment Metrics</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div className="text-muted-foreground">Optical Keypoints:</div>
                      <div className="font-medium">{metrics.keypointsOptical}</div>
                      
                      <div className="text-muted-foreground">Thermal Keypoints:</div>
                      <div className="font-medium">{metrics.keypointsThermal}</div>
                      
                      <div className="text-muted-foreground">Initial Matches:</div>
                      <div className="font-medium">{metrics.matches}</div>
                      
                      <div className="text-muted-foreground">RANSAC Inliers:</div>
                      <div className="font-medium text-green-500">{metrics.inliers}</div>
                      
                      <div className="text-muted-foreground">Alignment RMSE:</div>
                      <div className="font-medium">{metrics.rmse} px</div>
                    </div>
                    <Badge variant="outline" className="w-full justify-center">
                      <Crosshair className="mr-1 h-3 w-3" />
                      High Quality Alignment
                    </Badge>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Visualization */}
            <div className="lg:col-span-2">
              <Card className="h-full">
                <CardHeader>
                  <CardTitle>Alignment Visualization</CardTitle>
                  <CardDescription>Keypoint matches and transformation preview</CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs defaultValue="matches" className="w-full">
                    <TabsList className="grid w-full grid-cols-3">
                      <TabsTrigger value="matches">Keypoint Matches</TabsTrigger>
                      <TabsTrigger value="overlay">Overlay</TabsTrigger>
                      <TabsTrigger value="difference">Difference Map</TabsTrigger>
                    </TabsList>
                    <TabsContent value="matches" className="mt-4">
                      <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center text-muted-foreground">
                          <Crosshair className="h-12 w-12 mx-auto mb-2" />
                          <p>Keypoint matches visualization</p>
                          {aligned && <p className="text-sm mt-1">{metrics.inliers} inlier matches shown</p>}
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="overlay" className="mt-4">
                      <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center text-muted-foreground">
                          <p>Optical + Thermal overlay</p>
                          {aligned && <p className="text-sm mt-1 text-green-500">Aligned successfully</p>}
                        </div>
                      </div>
                    </TabsContent>
                    <TabsContent value="difference" className="mt-4">
                      <div className="aspect-video bg-muted rounded-lg flex items-center justify-center">
                        <div className="text-center text-muted-foreground">
                          <p>Difference heatmap</p>
                          {aligned && <p className="text-sm mt-1">RMSE: {metrics.rmse} pixels</p>}
                        </div>
                      </div>
                    </TabsContent>
                  </Tabs>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button variant="outline" asChild>
              <Link href="/ingest">Back to Ingest</Link>
            </Button>
            <Button
              disabled={!aligned}
              asChild
              className="bg-[#0E88D3] hover:bg-[#0E88D3]/90"
            >
              <Link href="/fuse-sr">
                Continue to Fusion & SR
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </Button>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}