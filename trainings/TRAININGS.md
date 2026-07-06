# Bitácora de entrenamientos

Registro automático (lo escriben los scripts de entrenamiento). Fila por corrida, clave = `id`. Estados: `en_curso`, `hecho`, `fallido`.

| id                             | modelo                 | datos                        | estado   | épocas | val    | test   | checkpoint                                                | actualizado      |
| ------------------------------ | ---------------------- | ---------------------------- | -------- | ------ | ------ | ------ | --------------------------------------------------------- | ---------------- |
| noise_compare_limpio           | CNN[16,32,64] d0.2     | mnist_limpio                 | hecho    | 20/20  | 0.9935 | 0.9939 | experiments/_noise_compare/ckpt_clean.pt                  | 2026-07-05 16:24 |
| noise_compare_ruidoso          | CNN[16,32,64] d0.2     | gaussiano nivel_3            | hecho    | 20/20  | 0.9832 | 0.9823 | experiments/_noise_compare/ckpt_noisy.pt                  | 2026-07-05 16:24 |
| kernel_k3                      | CNN[16,32,64] k=3 d0.2 | mnist_limpio                 | hecho    | 15/15  | 0.9932 | 0.9937 | experiments/_kernel_sweep/ckpt_k3.pt                      | 2026-07-05 16:49 |
| kernel_k5                      | CNN[16,32,64] k=5 d0.2 | mnist_limpio                 | hecho    | 15/15  | 0.9891 | 0.9883 | experiments/_kernel_sweep/ckpt_k5.pt                      | 2026-07-05 16:49 |
| kernel_k7                      | CNN[16,32,64] k=7 d0.2 | mnist_limpio                 | hecho    | 15/15  | 0.9838 | 0.9852 | experiments/_kernel_sweep/ckpt_k7.pt                      | 2026-07-05 16:49 |
| noisetype_limpio               | CNN[16, 32, 64] d0.2   | mnist_limpio                 | en_curso | 2/3    | 0.9852 | —      | experiments\_noisetype_sweep\ckpt_limpio.pt               | 2026-07-05 19:22 |
| noisetype_gaussiano            | CNN[16, 32, 64] d0.2   | gaussiano nivel_2            | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_gaussiano.pt            | 2026-07-05 19:21 |
| noisetype_sal_pimienta         | CNN[16, 32, 64] d0.2   | sal_pimienta nivel_2         | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_sal_pimienta.pt         | 2026-07-05 19:21 |
| noisetype_speckle              | CNN[16, 32, 64] d0.2   | speckle nivel_2              | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_speckle.pt              | 2026-07-05 19:21 |
| noisetype_poisson              | CNN[16, 32, 64] d0.2   | poisson nivel_2              | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_poisson.pt              | 2026-07-05 19:21 |
| noisetype_uniforme             | CNN[16, 32, 64] d0.2   | uniforme nivel_2             | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_uniforme.pt             | 2026-07-05 19:21 |
| noisetype_desenfoque_gaussiano | CNN[16, 32, 64] d0.2   | desenfoque_gaussiano nivel_2 | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_desenfoque_gaussiano.pt | 2026-07-05 19:21 |
| noisetype_oclusion             | CNN[16, 32, 64] d0.2   | oclusion nivel_2             | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_oclusion.pt             | 2026-07-05 19:21 |
| noisetype_iluminacion_desigual | CNN[16, 32, 64] d0.2   | iluminacion_desigual nivel_2 | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_iluminacion_desigual.pt | 2026-07-05 19:21 |
| noisetype_distorsion_elastica  | CNN[16, 32, 64] d0.2   | distorsion_elastica nivel_2  | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_distorsion_elastica.pt  | 2026-07-05 19:21 |
| noisetype_rayas_horizontales   | CNN[16, 32, 64] d0.2   | rayas_horizontales nivel_2   | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_rayas_horizontales.pt   | 2026-07-05 19:21 |
| noisetype_cuantizacion         | CNN[16, 32, 64] d0.2   | cuantizacion nivel_2         | en_curso | 0/3    | —      | —      | experiments\_noisetype_sweep\ckpt_cuantizacion.pt         | 2026-07-05 19:21 |

> Generado automáticamente por `nnist.utils.trainlog`. No editar a mano mientras haya corridas activas (se reescribe entero en cada actualización).
