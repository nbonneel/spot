/// author: Nicolas Bonneel (nbonneel@seas.harvard.edu)
// small helper for SSE and AVX
/*
  Copyright (c) 2019 CNRS
  Nicolas Bonneel <nicolas.bonneel@liris.cnrs.fr>
  David Coeurjolly <david.coeurjolly@liris.cnrs.fr>

  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are met:

  1. Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  2. Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIEDi
  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once
#if defined(__linux__) || defined(WIN32)
#include <malloc.h>
#endif
#include <intrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <bitset>
#include <string>
#include <iostream>
#include <vector>
#include <array>

#ifndef _MSC_VER
#include <immintrin.h>
#endif

#ifdef AVX_SUPPORT
#define SSE
#endif

#ifdef _MSC_VER
class InstructionSet
{
    // forward declarations
    class InstructionSet_Internal;

public:
    // getters
    static std::string Vendor(void) { return CPU_Rep.vendor_; }
    static std::string Brand(void) { return CPU_Rep.brand_; }

    static bool SSE3(void) { return CPU_Rep.f_1_ECX_[0]; }
    static bool PCLMULQDQ(void) { return CPU_Rep.f_1_ECX_[1]; }
    static bool MONITOR(void) { return CPU_Rep.f_1_ECX_[3]; }
    static bool SSSE3(void) { return CPU_Rep.f_1_ECX_[9]; }
    static bool FMA(void) { return CPU_Rep.f_1_ECX_[12]; }
    static bool CMPXCHG16B(void) { return CPU_Rep.f_1_ECX_[13]; }
    static bool SSE41(void) { return CPU_Rep.f_1_ECX_[19]; }
    static bool SSE42(void) { return CPU_Rep.f_1_ECX_[20]; }
    static bool MOVBE(void) { return CPU_Rep.f_1_ECX_[22]; }
    static bool POPCNT(void) { return CPU_Rep.f_1_ECX_[23]; }
    static bool AES(void) { return CPU_Rep.f_1_ECX_[25]; }
    static bool XSAVE(void) { return CPU_Rep.f_1_ECX_[26]; }
    static bool OSXSAVE(void) { return CPU_Rep.f_1_ECX_[27]; }
    static bool AVX(void) { return CPU_Rep.f_1_ECX_[28]; }
    static bool F16C(void) { return CPU_Rep.f_1_ECX_[29]; }
    static bool RDRAND(void) { return CPU_Rep.f_1_ECX_[30]; }

    static bool MSR(void) { return CPU_Rep.f_1_EDX_[5]; }
    static bool CX8(void) { return CPU_Rep.f_1_EDX_[8]; }
    static bool SEP(void) { return CPU_Rep.f_1_EDX_[11]; }
    static bool CMOV(void) { return CPU_Rep.f_1_EDX_[15]; }
    static bool CLFSH(void) { return CPU_Rep.f_1_EDX_[19]; }
    static bool MMX(void) { return CPU_Rep.f_1_EDX_[23]; }
    static bool FXSR(void) { return CPU_Rep.f_1_EDX_[24]; }
    static bool SSE(void) { return CPU_Rep.f_1_EDX_[25]; }
    static bool SSE2(void) { return CPU_Rep.f_1_EDX_[26]; }

    static bool FSGSBASE(void) { return CPU_Rep.f_7_EBX_[0]; }
    static bool BMI1(void) { return CPU_Rep.f_7_EBX_[3]; }
    static bool HLE(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[4]; }
    static bool AVX2(void) { return CPU_Rep.f_7_EBX_[5]; }
    static bool BMI2(void) { return CPU_Rep.f_7_EBX_[8]; }
    static bool ERMS(void) { return CPU_Rep.f_7_EBX_[9]; }
    static bool INVPCID(void) { return CPU_Rep.f_7_EBX_[10]; }
    static bool RTM(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_7_EBX_[11]; }
    static bool AVX512F(void) { return CPU_Rep.f_7_EBX_[16]; }
    static bool RDSEED(void) { return CPU_Rep.f_7_EBX_[18]; }
    static bool ADX(void) { return CPU_Rep.f_7_EBX_[19]; }
    static bool AVX512PF(void) { return CPU_Rep.f_7_EBX_[26]; }
    static bool AVX512ER(void) { return CPU_Rep.f_7_EBX_[27]; }
    static bool AVX512CD(void) { return CPU_Rep.f_7_EBX_[28]; }
    static bool SHA(void) { return CPU_Rep.f_7_EBX_[29]; }

    static bool PREFETCHWT1(void) { return CPU_Rep.f_7_ECX_[0]; }

    static bool LAHF(void) { return CPU_Rep.f_81_ECX_[0]; }
    static bool LZCNT(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_ECX_[5]; }
    static bool ABM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[5]; }
    static bool SSE4a(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[6]; }
    static bool XOP(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[11]; }
    static bool TBM(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_ECX_[21]; }

    static bool SYSCALL(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[11]; }
    static bool MMXEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[22]; }
    static bool RDTSCP(void) { return CPU_Rep.isIntel_ && CPU_Rep.f_81_EDX_[27]; }
    static bool _3DNOWEXT(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[30]; }
    static bool _3DNOW(void) { return CPU_Rep.isAMD_ && CPU_Rep.f_81_EDX_[31]; }

private:
    static  const InstructionSet_Internal CPU_Rep;

    class InstructionSet_Internal
    {
    public:
        InstructionSet_Internal()
            : nIds_{ 0 },
            nExIds_{ 0 },
            isIntel_{ false },
            isAMD_{ false },
            f_1_ECX_{ 0 },
            f_1_EDX_{ 0 },
            f_7_EBX_{ 0 },
            f_7_ECX_{ 0 },
            f_81_ECX_{ 0 },
            f_81_EDX_{ 0 },
            data_{},
            extdata_{}
        {
            //int cpuInfo[4] = {-1};
            std::array<int, 4> cpui;

            // Calling __cpuid with 0x0 as the function_id argument
            // gets the number of the highest valid function ID.
            __cpuid(cpui.data(), 0);
            nIds_ = cpui[0];

            for (int i = 0; i <= nIds_; ++i)
            {
                __cpuidex(cpui.data(), i, 0);
                data_.push_back(cpui);
            }

            // Capture vendor string
            char vendor[0x20];
            memset(vendor, 0, sizeof(vendor));
            *reinterpret_cast<int*>(vendor) = data_[0][1];
            *reinterpret_cast<int*>(vendor + 4) = data_[0][3];
            *reinterpret_cast<int*>(vendor + 8) = data_[0][2];
            vendor_ = vendor;
            if (vendor_ == "GenuineIntel")
            {
                isIntel_ = true;
            }
            else if (vendor_ == "AuthenticAMD")
            {
                isAMD_ = true;
            }

            // load bitset with flags for function 0x00000001
            if (nIds_ >= 1)
            {
                f_1_ECX_ = data_[1][2];
                f_1_EDX_ = data_[1][3];
            }

            // load bitset with flags for function 0x00000007
            if (nIds_ >= 7)
            {
                f_7_EBX_ = data_[7][1];
                f_7_ECX_ = data_[7][2];
            }

            // Calling __cpuid with 0x80000000 as the function_id argument
            // gets the number of the highest valid extended ID.
            __cpuid(cpui.data(), 0x80000000);
            nExIds_ = cpui[0];

            char brand[0x40];
            memset(brand, 0, sizeof(brand));

            for (int i = 0x80000000; i <= nExIds_; ++i)
            {
                __cpuidex(cpui.data(), i, 0);
                extdata_.push_back(cpui);
            }

            // load bitset with flags for function 0x80000001
            if (nExIds_ >= 0x80000001)
            {
                f_81_ECX_ = extdata_[1][2];
                f_81_EDX_ = extdata_[1][3];
            }

            // Interpret CPU brand string if reported
            if (nExIds_ >= 0x80000004)
            {
                memcpy(brand, extdata_[2].data(), sizeof(cpui));
                memcpy(brand + 16, extdata_[3].data(), sizeof(cpui));
                memcpy(brand + 32, extdata_[4].data(), sizeof(cpui));
                brand_ = brand;
            }
        };

        int nIds_;
        int nExIds_;
        std::string vendor_;
        std::string brand_;
        bool isIntel_;
        bool isAMD_;
        std::bitset<32> f_1_ECX_;
        std::bitset<32> f_1_EDX_;
        std::bitset<32> f_7_EBX_;
        std::bitset<32> f_7_ECX_;
        std::bitset<32> f_81_ECX_;
        std::bitset<32> f_81_EDX_;
        std::vector<std::array<int, 4>> data_;
        std::vector<std::array<int, 4>> extdata_;
    };
};



static inline void disp_supported() {
	auto& outstream = std::cout;

	auto support_message = [&outstream](std::string isa_feature, bool is_supported) {
		outstream << isa_feature << (is_supported ? " supported" : " not supported") << std::endl;
	};

	std::cout << InstructionSet::Vendor() << std::endl;
	std::cout << InstructionSet::Brand() << std::endl;

	support_message("3DNOW", InstructionSet::_3DNOW());
	support_message("3DNOWEXT", InstructionSet::_3DNOWEXT());
	support_message("ABM", InstructionSet::ABM());
	support_message("ADX", InstructionSet::ADX());
	support_message("AES", InstructionSet::AES());
	support_message("AVX", InstructionSet::AVX());
	support_message("AVX2", InstructionSet::AVX2());
	support_message("AVX512CD", InstructionSet::AVX512CD());
	support_message("AVX512ER", InstructionSet::AVX512ER());
	support_message("AVX512F", InstructionSet::AVX512F());
	support_message("AVX512PF", InstructionSet::AVX512PF());
	support_message("BMI1", InstructionSet::BMI1());
	support_message("BMI2", InstructionSet::BMI2());
	support_message("CLFSH", InstructionSet::CLFSH());
	support_message("CMPXCHG16B", InstructionSet::CMPXCHG16B());
	support_message("CX8", InstructionSet::CX8());
	support_message("ERMS", InstructionSet::ERMS());
	support_message("F16C", InstructionSet::F16C());
	support_message("FMA", InstructionSet::FMA());
	support_message("FSGSBASE", InstructionSet::FSGSBASE());
	support_message("FXSR", InstructionSet::FXSR());
	support_message("HLE", InstructionSet::HLE());
	support_message("INVPCID", InstructionSet::INVPCID());
	support_message("LAHF", InstructionSet::LAHF());
	support_message("LZCNT", InstructionSet::LZCNT());
	support_message("MMX", InstructionSet::MMX());
	support_message("MMXEXT", InstructionSet::MMXEXT());
	support_message("MONITOR", InstructionSet::MONITOR());
	support_message("MOVBE", InstructionSet::MOVBE());
	support_message("MSR", InstructionSet::MSR());
	support_message("OSXSAVE", InstructionSet::OSXSAVE());
	support_message("PCLMULQDQ", InstructionSet::PCLMULQDQ());
	//support_message("POPCNT", InstructionSet::POPCNT());
	support_message("PREFETCHWT1", InstructionSet::PREFETCHWT1());
	support_message("RDRAND", InstructionSet::RDRAND());
	support_message("RDSEED", InstructionSet::RDSEED());
	support_message("RDTSCP", InstructionSet::RDTSCP());
	support_message("RTM", InstructionSet::RTM());
	support_message("SEP", InstructionSet::SEP());
	support_message("SHA", InstructionSet::SHA());
	support_message("SSE", InstructionSet::SSE());
	support_message("SSE2", InstructionSet::SSE2());
	support_message("SSE3", InstructionSet::SSE3());
	support_message("SSE4.1", InstructionSet::SSE41());
	support_message("SSE4.2", InstructionSet::SSE42());
	support_message("SSE4a", InstructionSet::SSE4a());
	support_message("SSSE3", InstructionSet::SSSE3());
	support_message("SYSCALL", InstructionSet::SYSCALL());
	support_message("TBM", InstructionSet::TBM());
	support_message("XOP", InstructionSet::XOP());
	support_message("XSAVE", InstructionSet::XSAVE());
}
#endif
static inline void * malloc_simd(const size_t size, const size_t alignment)
{
#if defined(WIN32) || defined(_MSC_VER)           // WIN32
    return _aligned_malloc(size, alignment);
#elif defined __linux__     // Linux
    return memalign(alignment, size);
#elif defined __MACH__      // Mac OS X
    return malloc(size);
#else                       // other (use valloc for page-aligned memory)
   // return valloc(size);
	return _aligned_malloc(size, alignment);
#endif
}

static inline void free_simd(void* mem)
{
#if defined(WIN32) || defined(_MSC_VER)           // WIN32
    return _aligned_free(mem);
#elif defined __linux__     // Linux
    free(mem);
#elif defined __MACH__      // Mac OS X
    free(mem);
#else                       // other (use valloc for page-aligned memory)
    free(mem);
#endif
}

static inline void *malloc_simd_offset(size_t bytes, size_t alignment, size_t offset)  // is such that variable+offset is aligned to alignment bytes (offset in bytes)
{
	void *p1, *p2; // basic pointer needed for computation.
	if ((p1 = (void *)malloc(bytes + alignment + offset + sizeof(size_t))) == NULL)
		return NULL;

	size_t addr = (size_t)p1 + alignment + sizeof(size_t) + offset;

	p2 = (void *)(addr - (addr%alignment) - offset);
	*((size_t *)p2 - 1) = (size_t)p1;

	return p2;
}

/************************************************
Name	 :- aligned_free
Arguments	:- pointer to be freed
Returns	 :- Nothing
*************************************************/

static inline void free_simd_offset(void *p)
{
	free((void *)(*((size_t *)p - 1)));
}


#ifdef SSE

#ifdef AVX_SUPPORT
#define ALIGN 32
#define VECSIZEDOUBLE 4
#define VECSIZEFLOAT 8
#define simd_add(x,y) _mm256_add_pd(x,y)
#define simd_sub(x,y) _mm256_sub_pd(x,y)
#define simd_mul(x,y) _mm256_mul_pd(x,y)
#define simd_div(x,y) _mm256_div_pd(x,y)
#define simd_max(x,y) _mm256_max_pd(x,y)
#define simd_load(x) _mm256_load_pd(x)
#define simd_store(x,y) _mm256_store_pd(x,y)
#define simd_set1(x) _mm256_set1_pd(x)
#define simd_or(x,y) _mm256_or_pd(x,y)
#define simd_gt(x,y) _mm256_cmp_pd(x,y,_CMP_GT_OS)
#define simd_and(x,y) _mm256_and_pd(x,y)
#define simd_andnot(x,y) _mm256_andnot_pd(x,y)

#define simd_add_f(x,y) _mm256_add_ps(x,y)
#define simd_sub_f(x,y) _mm256_sub_ps(x,y)
#define simd_mul_f(x,y) _mm256_mul_ps(x,y)
#define simd_div_f(x,y) _mm256_div_ps(x,y)
#define simd_max_f(x,y) _mm256_max_ps(x,y)
#define simd_load_f(x) _mm256_load_ps(x)
#define simd_store_f(x,y) _mm256_store_ps(x,y)
#define simd_set1_f(x) _mm256_set1_ps(x)
#define simd_or_f(x,y) _mm256_or_ps(x,y)
#define simd_gt_f(x,y) _mm256_cmp_ps(x,y,_CMP_GT_OS)
#define simd_and_f(x,y) _mm256_and_ps(x,y)
#define simd_andnot_f(x,y) _mm256_andnot_ps(x,y)

typedef __m256d simd_double;
typedef __m256 simd_float;

#else

#define ALIGN 16
#define VECSIZEDOUBLE 2
#define VECSIZEFLOAT 4
#define simd_add(x,y) _mm_add_pd(x,y)
#define simd_sub(x,y) _mm_sub_pd(x,y)
#define simd_mul(x,y) _mm_mul_pd(x,y)
#define simd_div(x,y) _mm_div_pd(x,y)
#define simd_max(x,y) _mm_max_pd(x,y)
#define simd_load(x) _mm_load_pd(x)
#define simd_store(x,y) _mm_store_pd(x,y)
#define simd_set1(x) _mm_set1_pd(x)
#define simd_or(x,y) _mm_or_pd(x,y)
#define simd_gt(x,y) _mm_cmpgt_pd(x,y)
#define simd_and(x,y) _mm_and_pd(x,y)
#define simd_andnot(x,y) _mm_andnot_pd(x,y)

#define simd_add_f(x,y) _mm_add_ps(x,y)
#define simd_sub_f(x,y) _mm_sub_ps(x,y)
#define simd_mul_f(x,y) _mm_mul_ps(x,y)
#define simd_div_f(x,y) _mm_div_ps(x,y)
#define simd_max_f(x,y) _mm_max_ps(x,y)
#define simd_load_f(x) _mm_load_ps(x)
#define simd_store_f(x,y) _mm_store_ps(x,y)
#define simd_set1_f(x) _mm_set1_ps(x)
#define simd_or_f(x,y) _mm_or_ps(x,y)
#define simd_gt_f(x,y) _mm_cmpgt_ps(x,y)
#define simd_and_f(x,y) _mm_and_ps(x,y)
#define simd_andnot_f(x,y) _mm_andnot_ps(x,y)

typedef __m128d simd_double;
typedef __m128 simd_float;
#endif

#endif
