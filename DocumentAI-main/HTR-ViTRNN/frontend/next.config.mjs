/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  // Disable linting and typechecking during production build since we check it during CI
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
