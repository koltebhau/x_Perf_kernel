cmd_arch/arm/mach-msm/idle-v7.o := arm-none-linux-gnueabi-gcc -Wp,-MD,arch/arm/mach-msm/.idle-v7.o.d  -nostdinc -isystem /usr/src/dropad/arm-2009q3/bin/../lib/gcc/arm-none-linux-gnueabi/4.4.1/include -I/usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include -Iinclude  -include include/generated/autoconf.h -D__KERNEL__ -mlittle-endian -Iarch/arm/mach-msm/include -D__ASSEMBLY__ -mabi=aapcs-linux -mno-thumb-interwork -funwind-tables  -D__LINUX_ARM_ARCH__=7 -march=armv7-a  -include asm/unified.h -msoft-float       -c -o arch/arm/mach-msm/idle-v7.o arch/arm/mach-msm/idle-v7.S

deps_arch/arm/mach-msm/idle-v7.o := \
  arch/arm/mach-msm/idle-v7.S \
    $(wildcard include/config/msm/cpu/avs.h) \
    $(wildcard include/config/msm/fiq/support.h) \
    $(wildcard include/config/nr/cpus.h) \
    $(wildcard include/config/msm/jtag/v7.h) \
  /usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include/asm/unified.h \
    $(wildcard include/config/arm/asm/unified.h) \
    $(wildcard include/config/thumb2/kernel.h) \
  include/linux/linkage.h \
  include/linux/compiler.h \
    $(wildcard include/config/trace/branch/profiling.h) \
    $(wildcard include/config/profile/all/branches.h) \
    $(wildcard include/config/enable/must/check.h) \
    $(wildcard include/config/enable/warn/deprecated.h) \
  /usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include/asm/linkage.h \
  include/linux/threads.h \
    $(wildcard include/config/base/small.h) \
  /usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include/asm/assembler.h \
    $(wildcard include/config/cpu/feroceon.h) \
    $(wildcard include/config/trace/irqflags.h) \
    $(wildcard include/config/smp.h) \
  /usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include/asm/ptrace.h \
    $(wildcard include/config/cpu/endian/be8.h) \
    $(wildcard include/config/arm/thumb.h) \
  /usr/src/dell/abc/dsc-team-kernel-project/arch/arm/include/asm/hwcap.h \

arch/arm/mach-msm/idle-v7.o: $(deps_arch/arm/mach-msm/idle-v7.o)

$(deps_arch/arm/mach-msm/idle-v7.o):
