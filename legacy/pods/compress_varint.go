package pods

func VarintEncode(u uint64) []byte {
	var b []byte
	for u >= 0x80 {
		b = append(b, byte(u)|0x80)
		u >>= 7
	}
	return append(b, byte(u))
}

func VarintDecode(b []byte) (u uint64, n int) {
	var shift uint
	for ; n < len(b); n++ {
		c := b[n]
		u |= uint64(c&0x7F) << shift
		if c&0x80 == 0 {
			n++
			break
		}
		shift += 7
	}
	return
}

func DeltaPack(in []uint64) []byte {
	out := make([]byte, 0, len(in))
	var prev uint64
	for i, v := range in {
		d := v
		if i > 0 {
			d = v - prev
		}
		out = append(out, VarintEncode(d)...)
		prev = v
	}
	return out
}
