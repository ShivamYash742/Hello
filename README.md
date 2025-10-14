# ISRO Thermal SR Lab

**Optics-Guided, Physics-Grounded Thermal Super-Resolution**

A production-ready Next.js application for thermal imagery super-resolution using optical guidance and physics-based constraints.

## Features

### 🚀 Complete Pipeline
- **Data Ingestion**: Upload HR optical (RGB GeoTIFF) and LR thermal IR imagery
- **Multi-Sensor Alignment**: Automatic feature-based registration with ORB/AKAZE/SIFT/SURF
- **Fusion & Super-Resolution**: 2×-4× upsampling with three model options (CNN, Guidance-Disentanglement, Swin Transformer)
- **Physics Constraints**: Atmospheric correction, emissivity handling, energy-balance regularization
- **Comprehensive Evaluation**: PSNR, SSIM, RMSE(K) metrics with per-class analysis
- **Interactive Map**: Layer visualization with opacity controls and comparison tools
- **Job Queue**: Batch processing with progress tracking

### 🎨 Modern UI
- **ISRO Brand Colors**: Rich Electric Blue (#0E88D3) and Pumpkin (#F47216)
- **Semi-transparent Navbar**: 80-90% opacity with backdrop blur
- **Dark Mode**: System-aware theme with persistent preferences
- **Volt-style Components**: Consistent dashboard design patterns
- **Responsive**: Mobile-first design with accessible focus rings

### 🔐 Authentication
- **PostgreSQL + Prisma**: Production-ready database setup
- **Credential Auth**: Email/password signup and login
- **Session Management**: Secure HTTP-only cookies
- **Protected Routes**: Middleware-based route protection

### 🛠️ Tech Stack
- **Frontend**: Next.js 14 (App Router), React 18, TailwindCSS, shadcn/ui
- **Backend**: Next.js API Routes, Prisma ORM
- **Database**: PostgreSQL with Prisma Accelerate
- **Auth**: Custom session-based authentication
- **UI**: Radix UI primitives, Lucide icons, next-themes

## Quick Start

### Prerequisites
- Node.js 18+
- PostgreSQL database (local or hosted)
- yarn package manager

### Installation

1. **Clone and install dependencies**:
```bash
cd /app
yarn install
```

2. **Set up environment variables**:
```bash
# .env file is already configured with:
DATABASE_URL="prisma+postgres://..."  # Your Prisma Accelerate URL
AUTH_SECRET="..."                      # Auth secret key
NEXTAUTH_URL="https://..."             # Your app URL
```

3. **Run database migrations**:
```bash
npx prisma generate
npx prisma db push
```

4. **Start development server**:
```bash
yarn dev
```

5. **Access the application**:
- Homepage: https://thermal-sr-optics.preview.emergentagent.com
- Login: https://thermal-sr-optics.preview.emergentagent.com/login
- Signup: https://thermal-sr-optics.preview.emergentagent.com/signup

## Application Routes

### Public Routes
- `/` - Overview with pipeline visualization
- `/about` - Technical approach and credits
- `/login` - User login
- `/signup` - User registration

### Protected Routes (require authentication)
- `/ingest` - Data upload and validation
- `/align` - Multi-sensor alignment controls
- `/fuse-sr` - Fusion and super-resolution processing
- `/physics` - Physics-based constraint configuration
- `/evaluate` - Metrics and quality assessment
- `/map` - Interactive map viewer
- `/jobs` - Job queue and status monitoring
- `/api-docs` - REST API documentation
- `/settings` - Application preferences

## API Endpoints

All API endpoints require authentication via session cookies.

### Authentication
- `POST /api/auth/signup` - Create new account
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout
- `GET /api/auth/me` - Get current user

### Data Processing
- `POST /api/upload` - Upload optical and thermal files
- `POST /api/jobs/{id}/align` - Start alignment
- `POST /api/jobs/{id}/sr` - Start super-resolution
- `GET /api/jobs/{id}` - Get job status
- `GET /api/jobs/{id}/metrics` - Get evaluation metrics
- `GET /api/jobs` - List all user jobs

## Database Schema

### User
- id, email, password (hashed), name, emailVerified
- Relationships: sessions, jobs

### Job
- id, userId, name, status, stage, progress
- Files: opticalFile, thermalFile, alignedFile, srFile, metricsFile
- config (JSON), errorMessage

### Session
- id, sessionToken, userId, expires

## Development

### File Structure
```
/app
├── app/                    # Next.js app directory
│   ├── (auth)/            # Auth pages (login, signup)
│   ├── api/               # API routes
│   ├── layout.js          # Root layout with theme provider
│   └── page.js            # Homepage
├── components/
│   ├── ui/                # shadcn/ui components
│   ├── navbar.js          # Main navigation
│   ├── dashboard-layout.js # Protected layout wrapper
│   └── theme-provider.js  # Dark mode provider
├── lib/
│   ├── auth.js            # Auth utilities
│   ├── prisma.js          # Prisma client
│   └── utils.js           # Utility functions
├── prisma/
│   └── schema.prisma      # Database schema
└── middleware.js          # Route protection

```

### Key Features Implementation

**Authentication Flow**:
1. User signs up → password hashed with bcrypt → user created in DB
2. Session created with 30-day expiry → session token stored in HTTP-only cookie
3. Protected routes check session via middleware → redirect to login if invalid

**Processing Pipeline** (Mocked for Demo):
1. Upload files → stored in `/uploads` directory → job created
2. Alignment → keypoint detection → RANSAC homography → aligned output
3. Fusion & SR → model selection → guided upsampling → SR output
4. Physics constraints → atmospheric correction → emissivity handling
5. Evaluation → compute metrics → generate visualizations

**Dark Mode**:
- Uses `next-themes` with class strategy
- System preference detection
- Persistent theme across reloads
- WCAG AA contrast in both themes

## Production Deployment

### Environment Variables Required
```env
DATABASE_URL="prisma+postgres://..."
AUTH_SECRET="<generate-with-openssl-rand-base64-32>"
NEXTAUTH_URL="https://your-domain.com"
```

### Database Setup
1. Create PostgreSQL database
2. Update DATABASE_URL in .env
3. Run: `npx prisma db push`
4. Verify tables: User, Account, Session, VerificationToken, Job

### Build for Production
```bash
yarn build
yarn start
```

## Architecture Decisions

1. **PostgreSQL over MongoDB**: Better relational data handling for jobs and user relationships
2. **Session-based Auth**: Simpler than JWT for server-rendered pages, secure HTTP-only cookies
3. **Prisma Accelerate**: Connection pooling for serverless/edge deployments
4. **Mocked ML Models**: UI/UX demonstration without GPU infrastructure
5. **Next.js App Router**: Modern React Server Components with streaming

## Use Cases

- **Urban Planning**: High-resolution thermal mapping for urban heat island analysis
- **Wildfire Monitoring**: Precise temperature mapping for early fire detection
- **Precision Agriculture**: Crop stress analysis with enhanced thermal detail
- **Industrial Inspection**: Equipment temperature monitoring at fine scales

## Credits

Developed as part of ISRO's thermal imaging research program.

**Technologies**: Next.js, React, Prisma, PostgreSQL, TailwindCSS, shadcn/ui, Radix UI

**License**: MIT

---

© 2025 ISRO Thermal SR Lab. Built with Next.js, Prisma, and PostgreSQL.
