import { compare, hash } from 'bcryptjs';
import prisma from './prisma';

export async function hashPassword(password) {
  return await hash(password, 12);
}

export async function verifyPassword(password, hashedPassword) {
  return await compare(password, hashedPassword);
}

export async function createUser(email, password, name) {
  const existingUser = await prisma.user.findUnique({
    where: { email }
  });

  if (existingUser) {
    throw new Error('User already exists');
  }

  const hashedPassword = await hashPassword(password);

  const user = await prisma.user.create({
    data: {
      email,
      password: hashedPassword,
      name: name || email.split('@')[0],
      emailVerified: new Date()
    }
  });

  return user;
}

export async function getUserByEmail(email) {
  return await prisma.user.findUnique({
    where: { email }
  });
}

export async function getSession(sessionToken) {
  const session = await prisma.session.findUnique({
    where: { sessionToken },
    include: { user: true }
  });

  if (!session || session.expires < new Date()) {
    return null;
  }

  return session;
}

export async function createSession(userId) {
  const sessionToken = crypto.randomUUID();
  const expires = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000); // 30 days

  const session = await prisma.session.create({
    data: {
      sessionToken,
      userId,
      expires
    },
    include: { user: true }
  });

  return session;
}

export async function deleteSession(sessionToken) {
  await prisma.session.delete({
    where: { sessionToken }
  });
}