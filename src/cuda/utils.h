// Utils for checking tensor properties and CUDA error handling
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Util for checking CUDA errors
#define CHECK_CUDA_ERROR(err) do { cuda_check((err), __FILE__, __LINE__); } while(false)
inline void cuda_check(cudaError_t error_code, const char *file, int line)
{
    if (error_code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error %d: %s. In file '%s' on line %d\n", error_code, cudaGetErrorString(error_code), file, line);
        fflush(stderr);
        exit(error_code);
    }
}

// Helper to check if a value is present in a list of compile-time constants.
template<auto V, auto... Vs>
struct contains : std::bool_constant<((V == Vs) || ...)> {};

// Recursive kernel dispatcher.
template <
    typename Launcher,
    size_t DimIndex = 0,
    typename SupportedDimsTuple,
    typename RuntimeDimsTuple,
    typename... MatchedDims
>
void dispatch_kernel(
    const Launcher& launcher,
    const RuntimeDimsTuple& runtime_dims,
    const SupportedDimsTuple& supported_dims,
    MatchedDims... matched_dims // Accumulator for matched std::integral_constant types
) {
    // Base case: All dimensions have been matched and accumulated.
    if constexpr (DimIndex == std::tuple_size_v<SupportedDimsTuple>) {
        // Launch the kernel with the accumulated compile-time types.
        launcher(matched_dims...);
    } else {
        // Recursive step: process the current dimension.
        const int runtime_value = std::get<DimIndex>(runtime_dims);
        const auto& supported_values_for_dim = std::get<DimIndex>(supported_dims);
        bool dispatched = false;

        // Unpack the supported values for the current dimension and find a match.
        std::apply([&](auto... compile_time_options) {
            // Use a fold expression to iterate through the compile-time options.
            ([&] {
                if (!dispatched && runtime_value == decltype(compile_time_options)::value) {
                    // Match found! Recurse to the next dimension, adding the
                    // matched type to our accumulator pack.
                    dispatch_kernel<Launcher, DimIndex + 1>(
                        launcher,
                        runtime_dims,
                        supported_dims,
                        matched_dims...,      // Pass along previously matched types
                        compile_time_options // Add the newly matched type
                    );
                    dispatched = true;
                }
            }(), ...);
        }, supported_values_for_dim);

        if (!dispatched) {
            TORCH_CHECK(false, "Unsupported value for dimension ", DimIndex, ": ", runtime_value);
        }
    }
}

// Kernel dispatcher with default fallback to dynamic kernel if no match found.
template <
    typename Launcher,
    typename DynamicLauncher,
    size_t DimIndex = 0,
    typename SupportedDimsTuple,
    typename RuntimeDimsTuple,
    typename... MatchedDims
>
void dispatch_kernel_with_fallback(
    const Launcher& launcher,
    const DynamicLauncher& dynamic_launcher,
    const RuntimeDimsTuple& runtime_dims,
    const SupportedDimsTuple& supported_dims,
    MatchedDims... matched_dims
) {
    if constexpr (DimIndex == std::tuple_size_v<SupportedDimsTuple>) {
        launcher(matched_dims...);
    } else {
        const int runtime_value = std::get<DimIndex>(runtime_dims);
        const auto& supported_values_for_dim = std::get<DimIndex>(supported_dims);
        bool dispatched = false;
        
        std::apply([&](auto... compile_time_options) {
            ([&] {
                if (!dispatched && runtime_value == decltype(compile_time_options)::value) {
                    dispatch_kernel_with_fallback<Launcher, DynamicLauncher, DimIndex + 1>(
                        launcher,
                        dynamic_launcher,
                        runtime_dims,
                        supported_dims,
                        matched_dims...,
                        compile_time_options
                    );
                    dispatched = true;
                }
            }(), ...);
        }, supported_values_for_dim);
        
        // If no template match found, use dynamic kernel
        if (!dispatched) {
            dynamic_launcher();
        }
    }
}