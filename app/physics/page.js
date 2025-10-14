'use client';

import { useState } from 'react';
import { DashboardLayout } from '@/components/dashboard-layout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { toast } from 'sonner';
import Link from 'next/link';
import { ArrowRight, Shield, Thermometer } from 'lucide-react';

export default function PhysicsPage() {
  const [atmosphericCorrection, setAtmosphericCorrection] = useState(true);
  const [emissivityMode, setEmissivityMode] = useState('fixed');
  const [emissivityValue, setEmissivityValue] = useState([0.95]);
  const [energyBalance, setEnergyBalance] = useState(true);
  const [thermalFidelity, setThermalFidelity] = useState([0.8]);
  const [sensorBand, setSensorBand] = useState('lwir');

  const handleApply = () => {
    toast.success('Physics constraints applied!');
  };

  return (
    <DashboardLayout>
      <div className="container mx-auto px-4 py-8">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Header */}
          <div>
            <h1 className="text-3xl font-bold mb-2 flex items-center gap-2">
              <Shield className="h-8 w-8 text-[#0E88D3]" />
              Physical Consistency
            </h1>
            <p className="text-muted-foreground">
              Configure radiative transfer, emissivity, and energy-balance constraints
            </p>
          </div>

          {/* Atmospheric Correction */}
          <Card>
            <CardHeader>
              <CardTitle>Atmospheric Correction</CardTitle>
              <CardDescription>Compensate for atmospheric absorption and scattering</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="atm-correction">Enable Atmospheric Correction</Label>
                <Switch
                  id="atm-correction"
                  checked={atmosphericCorrection}
                  onCheckedChange={setAtmosphericCorrection}
                />
              </div>
              {atmosphericCorrection && (
                <div className="space-y-4 pl-4 border-l-2 border-[#0E88D3]">
                  <div className="space-y-2">
                    <Label>Atmospheric Model</Label>
                    <Select defaultValue="midlatitude">
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="tropical">Tropical</SelectItem>
                        <SelectItem value="midlatitude">Mid-Latitude Summer</SelectItem>
                        <SelectItem value="subarctic">Subarctic</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label>Water Vapor (g/cm²)</Label>
                      <Input type="number" defaultValue="2.5" step="0.1" />
                    </div>
                    <div className="space-y-2">
                      <Label>Surface Pressure (mbar)</Label>
                      <Input type="number" defaultValue="1013" step="1" />
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Emissivity */}
          <Card>
            <CardHeader>
              <CardTitle>Emissivity Handling</CardTitle>
              <CardDescription>Configure surface emissivity for temperature retrieval</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label>Emissivity Mode</Label>
                <Select value={emissivityMode} onValueChange={setEmissivityMode}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fixed">Fixed Value</SelectItem>
                    <SelectItem value="landcover">Per Land Cover Class</SelectItem>
                    <SelectItem value="perpixel">Per-Pixel Map</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {emissivityMode === 'fixed' && (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <Label>Emissivity Value</Label>
                    <span className="text-sm text-muted-foreground">{emissivityValue[0].toFixed(2)}</span>
                  </div>
                  <Slider
                    value={emissivityValue}
                    onValueChange={setEmissivityValue}
                    min={0.85}
                    max={0.99}
                    step={0.01}
                  />
                </div>
              )}

              {emissivityMode === 'landcover' && (
                <div className="space-y-2 text-sm">
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>Urban:</span>
                      <span className="font-medium">0.92</span>
                    </div>
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>Vegetation:</span>
                      <span className="font-medium">0.98</span>
                    </div>
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>Water:</span>
                      <span className="font-medium">0.96</span>
                    </div>
                    <div className="flex justify-between p-2 bg-muted rounded">
                      <span>Bare Soil:</span>
                      <span className="font-medium">0.93</span>
                    </div>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Energy Balance */}
          <Card>
            <CardHeader>
              <CardTitle>Energy-Balance Regularization</CardTitle>
              <CardDescription>Constrain SR outputs to satisfy energy conservation</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex items-center justify-between">
                <Label htmlFor="energy-balance">Enable Energy-Balance Prior</Label>
                <Switch
                  id="energy-balance"
                  checked={energyBalance}
                  onCheckedChange={setEnergyBalance}
                />
              </div>
              {energyBalance && (
                <div className="space-y-2 pl-4 border-l-2 border-[#F47216]">
                  <p className="text-sm text-muted-foreground">
                    Applies Stefan-Boltzmann law to prevent unphysical temperature jumps across boundaries.
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Thermal Fidelity Guardrail */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Thermometer className="h-5 w-5 text-[#F47216]" />
                Thermal Fidelity Guardrail
              </CardTitle>
              <CardDescription>
                Limit optical texture leakage into temperature channels
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label>Fidelity Strength</Label>
                  <span className="text-sm text-muted-foreground">{thermalFidelity[0].toFixed(1)}</span>
                </div>
                <Slider
                  value={thermalFidelity}
                  onValueChange={setThermalFidelity}
                  min={0}
                  max={1}
                  step={0.1}
                />
                <p className="text-xs text-muted-foreground">
                  Higher values prevent optical artifacts at the cost of some detail loss
                </p>
              </div>
              <Badge variant="outline" className="w-full justify-center">
                {thermalFidelity[0] >= 0.7 ? 'High Fidelity' : thermalFidelity[0] >= 0.4 ? 'Balanced' : 'Detail Focused'}
              </Badge>
            </CardContent>
          </Card>

          {/* Sensor Configuration */}
          <Card>
            <CardHeader>
              <CardTitle>Sensor Band Settings</CardTitle>
              <CardDescription>Configure thermal sensor wavelength range</CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <Label>Thermal Band</Label>
              <Select value={sensorBand} onValueChange={setSensorBand}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="lwir">LWIR (8-14 µm)</SelectItem>
                  <SelectItem value="mwir">MWIR (3-5 µm)</SelectItem>
                </SelectContent>
              </Select>
            </CardContent>
          </Card>

          {/* Actions */}
          <div className="flex items-center justify-between">
            <Button variant="outline" asChild>
              <Link href="/fuse-sr">Back to Fusion & SR</Link>
            </Button>
            <Button
              onClick={handleApply}
              className="bg-[#0E88D3] hover:bg-[#0E88D3]/90"
            >
              Apply Constraints
              <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          </div>
        </div>
      </div>
    </DashboardLayout>
  );
}