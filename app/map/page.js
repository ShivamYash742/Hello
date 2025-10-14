'use client';

import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Map as MapIcon, Layers, Eye } from 'lucide-react';
import { useState } from 'react';

export default function MapPage() {
  const [opticalOpacity, setOpticalOpacity] = useState([100]);
  const [thermalOpacity, setThermalOpacity] = useState([70]);
  const [showOptical, setShowOptical] = useState(true);
  const [showThermal, setShowThermal] = useState(true);
  const [showSR, setShowSR] = useState(true);

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-7xl mx-auto space-y-4">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
              <MapIcon className="h-8 w-8 text-[#0E88D3]" />
              Interactive Map Viewer
            </h1>
            <p className="text-muted-foreground">
              Visualize optical, thermal, and SR outputs with layer controls
            </p>
          </div>

          <div className="grid lg:grid-cols-4 gap-4">
            {/* Layer Controls */}
            <div className="lg:col-span-1 space-y-4">
              <Card>
                <CardHeader>
                  <CardTitle className="text-base flex items-center gap-2">
                    <Layers className="h-4 w-4" />
                    Layer Controls
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-optical" className="text-sm">HR Optical</Label>
                      <Switch
                        id="show-optical"
                        checked={showOptical}
                        onCheckedChange={setShowOptical}
                      />
                    </div>
                    {showOptical && (
                      <div className="space-y-2 pl-4">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Opacity</span>
                          <span className="text-xs text-muted-foreground">{opticalOpacity[0]}%</span>
                        </div>
                        <Slider
                          value={opticalOpacity}
                          onValueChange={setOpticalOpacity}
                          min={0}
                          max={100}
                          step={5}
                        />
                      </div>
                    )}
                  </div>

                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <Label htmlFor="show-thermal" className="text-sm">LR Thermal</Label>
                      <Switch
                        id="show-thermal"
                        checked={showThermal}
                        onCheckedChange={setShowThermal}
                      />
                    </div>
                    {showThermal && (
                      <div className="space-y-2 pl-4">
                        <div className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">Opacity</span>
                          <span className="text-xs text-muted-foreground">{thermalOpacity[0]}%</span>
                        </div>
                        <Slider
                          value={thermalOpacity}
                          onValueChange={setThermalOpacity}
                          min={0}
                          max={100}
                          step={5}
                        />
                      </div>
                    )}
                  </div>

                  <div className="flex items-center justify-between">
                    <Label htmlFor="show-sr" className="text-sm">SR Output</Label>
                    <Switch
                      id="show-sr"
                      checked={showSR}
                      onCheckedChange={setShowSR}
                    />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Legend</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 bg-gradient-to-r from-blue-500 to-red-500 rounded" />
                    <span className="text-xs">Temperature (K)</span>
                  </div>
                  <div className="flex justify-between text-xs text-muted-foreground">
                    <span>280</span>
                    <span>290</span>
                    <span>300</span>
                    <span>310</span>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="text-base">Map Info</CardTitle>
                </CardHeader>
                <CardContent className="space-y-2 text-xs">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">CRS:</span>
                    <span className="font-medium">EPSG:4326</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Bounds:</span>
                    <span className="font-medium">28°N, 77°E</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Resolution:</span>
                    <span className="font-medium">10m/px</span>
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Map Viewport */}
            <div className="lg:col-span-3">
              <Card className="h-[calc(100vh-200px)]">
                <CardContent className="p-0 h-full">
                  <div className="w-full h-full bg-gradient-to-br from-green-500/10 via-yellow-500/10 to-red-500/10 rounded-lg flex items-center justify-center relative">
                    <div className="text-center text-muted-foreground space-y-2">
                      <MapIcon className="h-16 w-16 mx-auto" />
                      <p className="font-medium">Interactive Map View</p>
                      <p className="text-sm">
                        Map visualization with optical, thermal, and SR layers
                      </p>
                      <Badge variant="outline">Leaflet integration ready</Badge>
                    </div>
                    
                    {/* Map Controls */}
                    <div className="absolute top-4 right-4 space-y-2">
                      <Button size="sm" variant="secondary">
                        <Eye className="h-4 w-4 mr-2" />
                        Compare
                      </Button>
                    </div>

                    {/* Coordinates */}
                    <div className="absolute bottom-4 left-4 bg-background/90 backdrop-blur-sm px-3 py-1 rounded-md text-xs">
                      Lat: 28.6139° N, Lon: 77.2090° E
                    </div>

                    {/* Scale */}
                    <div className="absolute bottom-4 right-4 bg-background/90 backdrop-blur-sm px-3 py-1 rounded-md text-xs">
                      Scale: 1:50,000
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}
