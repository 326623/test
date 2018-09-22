.code
  GetAVXSupportFlag proc
  mov eax, 1
  cpuid
  xor eax, eax
  bt ecx, 28
  adc eax, eax
  ret
  GetAVXSupportFlag endp
  end
