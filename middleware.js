import { NextResponse } from 'next/server';

export async function middleware(request) {
  const { pathname } = request.nextUrl;
  
  // Public routes
  const publicRoutes = ['/', '/login', '/signup', '/about'];
  const isPublicRoute = publicRoutes.some(route => pathname === route || pathname.startsWith('/api/auth'));
  
  if (isPublicRoute) {
    return NextResponse.next();
  }
  
  // Check for session cookie
  const sessionToken = request.cookies.get('session')?.value;
  
  if (!sessionToken && !pathname.startsWith('/login') && !pathname.startsWith('/signup')) {
    return NextResponse.redirect(new URL('/login', request.url));
  }
  
  return NextResponse.next();
}

export const config = {
  matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};