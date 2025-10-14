'use client';

import { useState } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { toast } from 'sonner';
import Link from 'next/link';
import { ArrowRight, Play, Sparkles, Zap } from 'lucide-react';

export default function FuseSRPage() {
  const [model, setModel] = useState('cnn');
  const [scale, setScale] = useState('2');
  const [edgeWeight, setEdgeWeight] = useState([0.7]);
  const [textureWeight, setTextureWeight] = useState([0.5]);
  const [tvRegularization, setTvRegularization] = useState([0.01]);
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [completed, setCompleted] = useState(false);

  const handleProcess = async () => {
    setProcessing(true);
    setProgress(0);
    
    // Simulate processing with progress
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setProcessing(false);
          setCompleted(true);
          toast.success('Super-resolution completed!');
          return 100;
        }
        return prev + 5;
      });
    }, 200);
  };

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-6xl mx-auto space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
              <Sparkles className="h-8 w-8 text-[#F47216]" />
              Fusion & Super-Resolution
            </h1>
            <p className="text-muted-foreground">
              Guided SR using optical edges/textures to upsample thermal by 2×-4×
            </p>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Controls */}
            <div className="lg:col-span-1 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>SR Configuration</CardTitle>
                  <CardDescription>Choose model and parameters</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-2">
                    <Label>SR Model</Label>
                    <Select value={model} onValueChange={setModel}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cnn">Alignment-Fusion CNN</SelectItem>
                        <SelectItem value="disentangle">Guidance-Disentanglement</SelectItem>
                        <SelectItem value="transformer">Swin Transformer</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      {model === 'cnn' && 'Fast, efficient for real-time processing'}
                      {model === 'disentangle' && 'Separates thermal and optical features'}
                      {model === 'transformer' && 'Highest quality, slower processing'}
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Upsampling Scale</Label>
                    <Select value={scale} onValueChange={setScale}>
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="2">2× (512 → 1024)</SelectItem>
                        <SelectItem value="4">4× (512 → 2048)</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Edge Weight</Label>
                      <span className="text-sm text-muted-foreground">{edgeWeight[0].toFixed(2)}</span>
                    </div>
                    <Slider
                      value={edgeWeight}
                      onValueChange={setEdgeWeight}
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="text-xs text-muted-foreground">Optical edge guidance strength</p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>Texture Weight</Label>
                      <span className="text-sm text-muted-foreground">{textureWeight[0].toFixed(2)}</span>
                    </div>
                    <Slider
                      value={textureWeight}
                      onValueChange={setTextureWeight}
                      min={0}
                      max={1}
                      step={0.1}
                    />
                    <p className="text-xs text-muted-foreground">Optical texture transfer</p>
                  </div>

                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <Label>TV Regularization</Label>
                      <span className="text-sm text-muted-foreground">{tvRegularization[0].toFixed(3)}</span>
                    </div>
                    <Slider
                      value={tvRegularization}
                      onValueChange={setTvRegularization}
                      min={0}
                      max={0.1}
                      step={0.001}
                    />
                    <p className="text-xs text-muted-foreground">Smoothness constraint</p>
                  </div>

                  <Button
                    onClick={handleProcess}
                    disabled={processing}
                    className="w-full bg-[#F47216] hover:bg-[#F47216]/90"
                  >
                    {processing ? (
                      <>
                        <Zap className="mr-2 h-4 w-4 animate-pulse" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Play className="mr-2 h-4 w-4" />
                        Run Super-Resolution
                      </>
                    )}
                  </Button>
                </CardContent>
              </Card>

              {/* Model Info */}
              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Model Details</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Architecture:</span>
                    <span className="font-medium">{model.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Parameters:</span>
                    <span className="font-medium">
                      {model === 'cnn' && '2.4M'}
                      {model === 'disentangle' && '5.1M'}
                      {model === 'transformer' && '12.8M'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Est. Time:</span>
                    <span className="font-medium">
                      {model === 'cnn' && '~5s'}
                      {model === 'disentangle' && '~12s'}
                      {model === 'transformer' && '~25s'}
                    </span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Preview */}
            <div className="lg:col-span-2 space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>Processing Preview</CardTitle>
                  <CardDescription>Real-time super-resolution output</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label className="text-sm">Input (LR Thermal)</Label>
                      <div className="aspect-square bg-gradient-to-br from-red-500/20 to-yellow-500/20 rounded-lg flex items-center justify-center border">
                        <span className="text-sm text-muted-foreground">512 × 512</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-sm">Output (SR Thermal)</Label>
                      <div className="aspect-square bg-gradient-to-br from-red-500/30 to-yellow-500/30 rounded-lg flex items-center justify-center border-2 border-[#F47216]">
                        <span className="text-sm font-medium">
                          {scale === '2' ? '1024 × 1024' : '2048 × 2048'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {processing && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Processing {scale}× upsampling...</span>
                        <span>{progress}%</span>
                      </div>
                      <Progress value={progress} />
                      <div className="text-xs text-muted-foreground">
                        Stage: {progress < 30 ? 'Feature extraction' : progress < 70 ? 'Guided fusion' : 'Refinement'}
                      </div>
                    </div>
                  )}

                  {completed && (
                    <Badge variant="outline" className="w-full justify-center py-2">
                      <Sparkles className="mr-2 h-4 w-4 text-[#F47216]" />
                      Super-resolution complete!
                    </Badge>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Quality Indicators</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center p-3 rounded-lg bg-muted">
                      <div className="text-2xl font-bold text-[#0E88D3]">28.4</div>
                      <div className="text-xs text-muted-foreground">PSNR (dB)</div>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-muted">
                      <div className="text-2xl font-bold text-[#0E88D3]">0.89</div>
                      <div className="text-xs text-muted-foreground">SSIM</div>
                    </div>
                    <div className="text-center p-3 rounded-lg bg-muted">
                      <div className="text-2xl font-bold text-[#F47216]">1.2K</div>
                      <div className="text-xs text-muted-foreground">RMSE</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button variant="outline" asChild>
              <Link href="/align">Back to Alignment</Link>
            </Button>
            <div className="flex gap-2">
              <Button variant="outline" asChild>
                <Link href="/physics">Physics Settings</Link>
              </Button>
              <Button
                disabled={!completed}
                asChild
                className="bg-[#0E88D3] hover:bg-[#0E88D3]/90"
              >
                <Link href="/evaluate">
                  Evaluate Results
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}