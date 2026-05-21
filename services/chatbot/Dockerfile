FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# NEXT_PUBLIC_* values are baked at build time by Next.js. Override via
# compose build.args (prod overlay sets this to https://<host>/api).
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

RUN npm run build

EXPOSE 3000

CMD ["npm", "start"]
