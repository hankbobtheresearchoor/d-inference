// Generates deterministic NaCl box test vectors for cross-language validation.
//
// Usage: cd coordinator && go run ./internal/e2e/testdata/gen-vectors
package main

import (
	"crypto/ecdh"
	"encoding/base64"
	"encoding/hex"
	"fmt"

	"golang.org/x/crypto/nacl/box"
)

func main() {
	// Fixed provider key pair (recipient)
	providerPrivBytes := [32]byte{
		0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
		0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
		0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
		0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20,
	}

	// Fixed ephemeral key pair (sender / coordinator)
	ephPrivBytes := [32]byte{
		0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8,
		0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf, 0xb0,
		0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8,
		0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf, 0xc0,
	}

	providerPub := derivePub(providerPrivBytes[:])
	ephPub := derivePub(ephPrivBytes[:])

	var providerPubArr, ephPubArr [32]byte
	copy(providerPubArr[:], providerPub)
	copy(ephPubArr[:], ephPub)

	// Fixed nonce
	nonce := [24]byte{
		0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
		0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
		0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
	}

	fmt.Println("// Golden NaCl box test vectors")
	fmt.Println("// Provider private key:", hex.EncodeToString(providerPrivBytes[:]))
	fmt.Println("// Provider public key: ", hex.EncodeToString(providerPub))
	fmt.Println("// Ephemeral private key:", hex.EncodeToString(ephPrivBytes[:]))
	fmt.Println("// Ephemeral public key: ", hex.EncodeToString(ephPub))
	fmt.Println("// Nonce:                ", hex.EncodeToString(nonce[:]))
	fmt.Println()

	testCases := []struct {
		name      string
		plaintext string
	}{
		{"hello", "hello from Go"},
		{"json", `{"model":"test","messages":[{"role":"user","content":"hi"}]}`},
		{"empty", ""},
		{"unicode", "こんにちは世界 🌍"},
	}

	for _, tc := range testCases {
		plainBytes := []byte(tc.plaintext)
		encrypted := box.Seal(nonce[:], plainBytes, &nonce, &providerPubArr, &ephPrivBytes)

		fmt.Printf("Vector: %s\n", tc.name)
		fmt.Printf("  ephemeral_pub_b64:  %s\n", base64.StdEncoding.EncodeToString(ephPub))
		fmt.Printf("  ciphertext_b64:     %s\n", base64.StdEncoding.EncodeToString(encrypted))
		fmt.Printf("  provider_priv_b64:  %s\n", base64.StdEncoding.EncodeToString(providerPrivBytes[:]))
		fmt.Printf("  plaintext:          %q\n", tc.plaintext)
		fmt.Println()
	}

	// Reverse: provider encrypts → coordinator decrypts
	responseEncrypted := box.Seal(nonce[:], []byte("response from provider"), &nonce, &ephPubArr, &providerPrivBytes)
	fmt.Println("Vector: reverse")
	fmt.Printf("  sender_pub_b64:     %s\n", base64.StdEncoding.EncodeToString(providerPub))
	fmt.Printf("  ciphertext_b64:     %s\n", base64.StdEncoding.EncodeToString(responseEncrypted))
	fmt.Printf("  recipient_priv_b64: %s\n", base64.StdEncoding.EncodeToString(ephPrivBytes[:]))
	fmt.Printf("  plaintext:          %q\n", "response from provider")
}

func derivePub(privBytes []byte) []byte {
	key, err := ecdh.X25519().NewPrivateKey(privBytes)
	if err != nil {
		panic(err)
	}
	return key.PublicKey().Bytes()
}
