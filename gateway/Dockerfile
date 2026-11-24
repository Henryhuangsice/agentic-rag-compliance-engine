FROM node:22-slim AS build
WORKDIR /src
COPY package.json package-lock.json tsconfig.base.json ./
COPY shared/package.json shared/
COPY gateway/package.json gateway/
COPY web/package.json web/
RUN npm ci
COPY shared shared
COPY gateway gateway
COPY web web
RUN npm run build && npm prune --omit=dev

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production HOST=0.0.0.0 PORT=8787 WEB_DIST=/app/web
COPY --from=build /src/node_modules ./node_modules
COPY --from=build /src/gateway/package.json ./gateway/
COPY --from=build /src/gateway/dist ./gateway/dist
COPY --from=build /src/web/dist ./web
USER node
EXPOSE 8787
HEALTHCHECK --interval=30s --timeout=5s \
  CMD node -e "fetch('http://127.0.0.1:8787/healthz').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"
CMD ["node", "gateway/dist/index.js"]
