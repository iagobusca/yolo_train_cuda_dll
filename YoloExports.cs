// YoloExports.cs
// DLL: Yolo_train.dll
// TargetFramework: net10.0-windows
//
// OBJETIVO: expor funções C (UnmanagedCallersOnly) para criar tarefa,
// carregar modelo, treinar e fazer predição usando IntptrMax.YoloSharp,
// SEM depender de YoloTask/YoloResult em tempo de compilação.

using System;
using System.Collections;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Reflection;
using System.Text;

namespace Yolo_train
{
    public static class YoloExports
    {
        // Instância global da tarefa YOLO (YoloSharp.YoloTask em runtime).
        private static object? _task;
        private static object? _lastResults;
        private static string? _lastError;

        // ========================= HELPERS NATIVOS =========================

        private static string PtrToString(IntPtr ptr)
        {
            return ptr == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringUTF8(ptr) ?? string.Empty);
        }

        private static IntPtr StringToCoTaskMemUtf8(string? s)
        {
            if (string.IsNullOrEmpty(s))
                return IntPtr.Zero;

            byte[] bytes = Encoding.UTF8.GetBytes(s);
            IntPtr mem = Marshal.AllocCoTaskMem(bytes.Length + 1);
            Marshal.Copy(bytes, 0, mem, bytes.Length);
            Marshal.WriteByte(mem, bytes.Length, 0); // terminador \0
            return mem;
        }

        private static void SetLastError(string message)
        {
            _lastError = message;
        }

        /// <summary>
        /// Tenta obter o Type de YoloSharp.YoloTask da DLL IntptrMax.YoloSharp.
        /// </summary>
        private static Type? GetYoloTaskType()
        {
            // Nome mais provável: namespace YoloSharp, assembly IntptrMax.YoloSharp
            var t = Type.GetType("YoloSharp.YoloTask, IntptrMax.YoloSharp", throwOnError: false);
            if (t != null) return t;

            // Fallbacks (caso o autor tenha mudado namespace/assembly)
            t = Type.GetType("IntptrMax.YoloSharp.YoloTask, IntptrMax.YoloSharp", throwOnError: false);
            if (t != null) return t;

            // Tenta em qualquer assembly carregado
            foreach (var asm in AppDomain.CurrentDomain.GetAssemblies())
            {
                t = asm.GetType("YoloSharp.YoloTask", throwOnError: false);
                if (t != null) return t;
            }

            return null;
        }

        // ===================================================================
        //  C EXPORTS
        // ===================================================================

        /// <summary>
        /// Retorna a última mensagem de erro (UTF-8 alocado com CoTaskMem).
        /// O chamador deve liberar com CoTaskMemFree.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "yt_get_last_error", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static IntPtr GetLastError()
        {
            return StringToCoTaskMemUtf8(_lastError ?? string.Empty);
        }

        /// <summary>
        /// Cria a tarefa YoloTask.
        /// Retorna 0 em sucesso, !=0 em erro.
        /// </summary>
        /// <param name="taskType">int (enum TaskType)</param>
        /// <param name="numClasses">número de classes</param>
        /// <param name="yoloType">int (enum YoloType)</param>
        /// <param name="deviceType">int (enum DeviceType)</param>
        /// <param name="yoloSize">int (enum YoloSize)</param>
        /// <param name="dtype">int (ScalarType/dtype conforme lib)</param>
        /// <param name="keyPointShape">>0 para keypoints, 0 se não usar</param>
        [UnmanagedCallersOnly(EntryPoint = "yt_create_task", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static int CreateTask(
            int taskType,
            int numClasses,
            int yoloType,
            int deviceType,
            int yoloSize,
            int dtype,
            int keyPointShape)
        {
            try
            {
                var yoloTaskType = GetYoloTaskType();
                if (yoloTaskType is null)
                {
                    SetLastError("Não foi possível localizar o tipo YoloSharp.YoloTask. Verifique se IntptrMax.YoloSharp está referenciado e presente ao lado da DLL.");
                    _task = null;
                    _lastResults = null;
                    return 1;
                }

                int[]? kpShape = null;
                if (keyPointShape > 0)
                {
                    // Ajuste se o seu modelo precisar de outro formato (ex.: [17,3]).
                    kpShape = new[] { keyPointShape };
                }

                // Chama o construtor via reflection.
                // Os ints serão convertidos automaticamente para enums pela Reflection.
                object? instance = Activator.CreateInstance(
                    yoloTaskType,
                    new object?[]
                    {
                        taskType,     // TaskType
                        numClasses,   // numberClass
                        yoloType,     // YoloType
                        deviceType,   // DeviceType
                        yoloSize,     // YoloSize
                        dtype,        // dtype (ScalarType/enum)
                        kpShape       // keyPointShape (int[] ou null)
                    });

                if (instance is null)
                {
                    SetLastError("Falha ao criar instância de YoloTask (Activator.CreateInstance retornou null).");
                    _task = null;
                    _lastResults = null;
                    return 2;
                }

                _task = instance;
                _lastResults = null;
                SetLastError(string.Empty);
                return 0;
            }
            catch (Exception ex)
            {
                _task = null;
                _lastResults = null;
                SetLastError("Exceção em yt_create_task: " + ex);
                return 99;
            }
        }

        /// <summary>
        /// Carrega modelo pré-treinado (.pt) para a tarefa atual.
        /// preTrainedModelPathPtr: ponteiro UTF-8 para caminho.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "yt_load_model", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static int LoadModel(IntPtr preTrainedModelPathPtr, [MarshalAs(UnmanagedType.I1)] bool skipNcNotEqualLayers)
        {
            try
            {
                if (_task is null)
                {
                    SetLastError("yt_load_model chamado antes de yt_create_task.");
                    return 2;
                }

                string path = PtrToString(preTrainedModelPathPtr);
                var t = _task.GetType();

                // Tenta assinatura (string, bool)
                var mi = t.GetMethod("LoadModel", new[] { typeof(string), typeof(bool) })
                         ?? t.GetMethod("LoadModel"); // fallback genérico

                if (mi is null)
                {
                    SetLastError("Método LoadModel não encontrado em YoloTask.");
                    return 3;
                }

                mi.Invoke(_task, new object?[] { path, skipNcNotEqualLayers });
                SetLastError(string.Empty);
                return 0;
            }
            catch (Exception ex)
            {
                SetLastError("Exceção em yt_load_model: " + ex);
                return 99;
            }
        }

        /// <summary>
        /// Treina o modelo.
        /// Todos os caminhos são UTF-8; ints seguem a doc da lib.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "yt_train", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static int Train(
            IntPtr rootPathPtr,
            IntPtr trainDataPathPtr,
            IntPtr valDataPathPtr,
            IntPtr outputPathPtr,
            int imageSize,
            int batchSize,
            int epochs,
            int imageProcessType)
        {
            try
            {
                if (_task is null)
                {
                    SetLastError("yt_train chamado antes de yt_create_task.");
                    return 2;
                }

                string rootPath      = PtrToString(rootPathPtr);
                string trainDataPath = PtrToString(trainDataPathPtr);
                string valDataPath   = PtrToString(valDataPathPtr);
                string outputPath    = PtrToString(outputPathPtr);

                var t  = _task.GetType();
                var mi = t.GetMethod("Train");
                if (mi is null)
                {
                    SetLastError("Método Train não encontrado em YoloTask.");
                    return 3;
                }

                // Os ints (imageSize, batchSize, epochs, imageProcessType) serão convertidos
                // para os tipos corretos (incluindo enums) pela Reflection.
                mi.Invoke(
                    _task,
                    new object?[]
                    {
                        rootPath,
                        trainDataPath,
                        valDataPath,
                        outputPath,
                        imageSize,
                        batchSize,
                        epochs,
                        imageProcessType
                    });

                SetLastError(string.Empty);
                return 0;
            }
            catch (Exception ex)
            {
                SetLastError("Exceção em yt_train: " + ex);
                return 99;
            }
        }

        /// <summary>
        /// Faz predição em uma imagem.
        /// Armazena internamente a lista retornada por ImagePredict.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "yt_predict_image", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static int PredictImage(
            IntPtr predictImagePtr,
            float predictThreshold,
            float iouThreshold)
        {
            try
            {
                if (_task is null)
                {
                    SetLastError("yt_predict_image chamado antes de yt_create_task.");
                    _lastResults = null;
                    return 2;
                }

                string imagePath = PtrToString(predictImagePtr);
                var t  = _task.GetType();
                var mi = t.GetMethod("ImagePredict");
                if (mi is null)
                {
                    SetLastError("Método ImagePredict não encontrado em YoloTask.");
                    _lastResults = null;
                    return 3;
                }

                var resultObj = mi.Invoke(
                    _task,
                    new object?[] { imagePath, predictThreshold, iouThreshold });

                _lastResults = resultObj;
                SetLastError(string.Empty);
                return 0;
            }
            catch (Exception ex)
            {
                _lastResults = null;
                SetLastError("Exceção em yt_predict_image: " + ex);
                return 99;
            }
        }

        /// <summary>
        /// Retorna o número de resultados da última predição.
        /// </summary>
        [UnmanagedCallersOnly(EntryPoint = "yt_get_last_result_count", CallConvs = new[] { typeof(CallConvCdecl) })]
        public static int GetLastResultCount()
        {
            try
            {
                if (_lastResults is null) return 0;

                if (_lastResults is ICollection coll)
                    return coll.Count;

                if (_lastResults is IEnumerable en)
                {
                    int c = 0;
                    foreach (var _ in en) c++;
                    return c;
                }

                return 0;
            }
            catch
            {
                return 0;
            }
        }
    }
}
