# Vendored frontend libraries

Byte-identical copies of each package's ESM build from the npm registry,
mapped to their bare specifiers by the importmap in `../index.html`. No CDN:
the floor must work on a LAN with no internet.

| File | Package | Source path in the package | Licence |
| --- | --- | --- | --- |
| `preact-10.29.8.module.js` | `preact@10.29.8` | `dist/preact.module.js` | MIT |
| `preact-hooks-10.29.8.module.js` | `preact@10.29.8` | `hooks/dist/hooks.module.js` | MIT |
| `htm-3.1.1.module.js` | `htm@3.1.1` | `dist/htm.module.js` | Apache-2.0 |
| `htm-preact-3.1.1.module.js` | `htm@3.1.1` | `preact/index.module.js` | Apache-2.0 |
| `signals-core-1.14.4.module.js` | `@preact/signals-core@1.14.4` | `dist/signals-core.module.js` | MIT |
| `signals-2.11.2.module.js` | `@preact/signals@2.11.2` | `dist/signals.module.js` | MIT |

The `.LICENSE` files are each package's own `LICENSE`. The builds'
`sourceMappingURL` comments point at maps that are not vendored; with
devtools open that is a harmless 404.

## Upgrading

```bash
npm pack preact@X htm@Y @preact/signals@Z @preact/signals-core@W
```

Copy the paths above out of each tarball under the new version's filename,
delete the old file, update the importmap in `../index.html` and this table,
all in one commit. `@preact/signals` imports `preact`, `preact/hooks` and
`@preact/signals-core`, so check its peer range when bumping any of them.
`test_web_app.py` fails if an importmap target is missing.
