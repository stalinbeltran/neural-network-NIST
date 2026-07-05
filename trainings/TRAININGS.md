# Bitácora de entrenamientos

Registro automático (lo escriben los scripts de entrenamiento). Fila por corrida, clave = `id`. Estados: `en_curso`, `hecho`, `fallido`.

| id                    | modelo                 | datos             | estado | épocas | val    | test   | checkpoint                               | actualizado      |
| --------------------- | ---------------------- | ----------------- | ------ | ------ | ------ | ------ | ---------------------------------------- | ---------------- |
| noise_compare_limpio  | CNN[16,32,64] d0.2     | mnist_limpio      | hecho  | 20/20  | 0.9935 | 0.9939 | experiments/_noise_compare/ckpt_clean.pt | 2026-07-05 16:24 |
| noise_compare_ruidoso | CNN[16,32,64] d0.2     | gaussiano nivel_3 | hecho  | 20/20  | 0.9832 | 0.9823 | experiments/_noise_compare/ckpt_noisy.pt | 2026-07-05 16:24 |
| kernel_k3             | CNN[16,32,64] k=3 d0.2 | mnist_limpio      | hecho  | 15/15  | 0.9932 | 0.9937 | experiments/_kernel_sweep/ckpt_k3.pt     | 2026-07-05 16:49 |
| kernel_k5             | CNN[16,32,64] k=5 d0.2 | mnist_limpio      | hecho  | 15/15  | 0.9891 | 0.9883 | experiments/_kernel_sweep/ckpt_k5.pt     | 2026-07-05 16:49 |
| kernel_k7             | CNN[16,32,64] k=7 d0.2 | mnist_limpio      | hecho  | 15/15  | 0.9838 | 0.9852 | experiments/_kernel_sweep/ckpt_k7.pt     | 2026-07-05 16:49 |

> Generado automáticamente por `nnist.utils.trainlog`. No editar a mano mientras haya corridas activas (se reescribe entero en cada actualización).
