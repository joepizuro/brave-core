/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "bat/ads/internal/security/crypto_util.h"

#include <openssl/digest.h>
#include <openssl/hkdf.h>
#include <openssl/sha.h>

#include <algorithm>

#include "base/base64.h"
#include "base/json/json_reader.h"
#include "base/rand_util.h"
#include "base/stl_util.h"
#include "base/strings/string_number_conversions.h"
#include "base/values.h"
#include "bat/ads/internal/logging.h"
#include "bat/ads/internal/security/key_pair_info.h"
#include "bat/ads/internal/string_util.h"
#include "tweetnacl.h"  // NOLINT
#include "wrapper.hpp"

namespace ads {
namespace security {

std::string Sign(const std::map<std::string, std::string>& headers,
                 const std::string& key_id,
                 const std::string& secret_key) {
  if (headers.empty() || key_id.empty() || secret_key.empty()) {
    return "";
  }

  std::string concatenated_header = "";
  std::string concatenated_message = "";

  unsigned int index = 0;
  for (const auto& header : headers) {
    if (index != 0) {
      concatenated_header += " ";
      concatenated_message += "\n";
    }

    concatenated_header += header.first;
    concatenated_message += header.first + ": " + header.second;

    index++;
  }

  std::vector<uint8_t> signed_message(crypto_sign_BYTES +
                                      concatenated_message.length());

  std::vector<uint8_t> raw_secret_key;
  if (!base::HexStringToBytes(secret_key, &raw_secret_key)) {
    return "";
  }

  // Resolving the following linter error breaks the build on Windows
  unsigned long long signed_message_length = 0;  // NOLINT
  crypto_sign(
      &signed_message.front(), &signed_message_length,
      reinterpret_cast<const unsigned char*>(concatenated_message.c_str()),
      concatenated_message.length(), &raw_secret_key.front());

  std::vector<uint8_t> signature(crypto_sign_BYTES);
  std::copy(signed_message.begin(), signed_message.begin() + crypto_sign_BYTES,
            signature.begin());

  return "keyId=\"" + key_id + "\",algorithm=\"" + crypto_sign_PRIMITIVE +
         "\",headers=\"" + concatenated_header + "\",signature=\"" +
         base::Base64Encode(signature) + "\"";
}

std::vector<uint8_t> Sha256Hash(const std::string& value) {
  if (value.empty()) {
    return {};
  }

  std::vector<uint8_t> sha256(SHA256_DIGEST_LENGTH);
  SHA256((uint8_t*)value.c_str(), value.length(), &sha256.front());
  return sha256;
}

KeyPairInfo GenerateSignKeyPair() {
  std::vector<uint8_t> public_key(crypto_box_PUBLICKEYBYTES);
  std::vector<uint8_t> secret_key(crypto_box_SECRETKEYBYTES);
  crypto_sign_keypair(&public_key.front(), &secret_key.front(), 0);

  KeyPairInfo key_pair;
  key_pair.public_key = public_key;
  key_pair.secret_key = secret_key;

  return key_pair;
}

KeyPairInfo GenerateBoxKeyPair() {
  std::vector<uint8_t> public_key(crypto_box_PUBLICKEYBYTES);
  std::vector<uint8_t> secret_key(crypto_box_SECRETKEYBYTES);
  crypto_box_keypair(&public_key.front(), &secret_key.front());

  KeyPairInfo key_pair;
  key_pair.public_key = public_key;
  key_pair.secret_key = secret_key;

  return key_pair;
}

std::vector<uint8_t> GenerateNonce() {
  std::vector<uint8_t> nonce(crypto_box_NONCEBYTES);
  base::RandBytes(&nonce.front(), nonce.size());

  return nonce;
}

std::vector<uint8_t> Encrypt(const std::vector<uint8_t>& plaintext,
                             const std::vector<uint8_t>& nonce,
                             const std::vector<uint8_t>& public_key,
                             const std::vector<uint8_t>& ephemeral_secret_key) {
  // API requires 32 leading zero-padding bytes
  std::vector<uint8_t> padded_plaintext = plaintext;
  padded_plaintext.insert(padded_plaintext.begin(), crypto_box_ZEROBYTES, 0);

  std::vector<uint8_t> ciphertext(padded_plaintext.size());
  crypto_box(&ciphertext.front(), &padded_plaintext.front(),
             padded_plaintext.size(), &nonce.front(), &public_key.front(),
             &ephemeral_secret_key.front());

  return ciphertext;
}

std::vector<uint8_t> Decrypt(const std::vector<uint8_t>& ciphertext,
                             const std::vector<uint8_t>& nonce,
                             const std::vector<uint8_t>& ephemeral_public_key,
                             const std::vector<uint8_t>& secret_key) {
  std::vector<uint8_t> padded_plaintext(ciphertext.size());
  crypto_box_open(&padded_plaintext.front(), &ciphertext.front(),
                  ciphertext.size(), &nonce.front(),
                  &ephemeral_public_key.front(), &secret_key.front());

  std::vector<uint8_t> plaintext(
      padded_plaintext.begin() + crypto_box_ZEROBYTES, padded_plaintext.end());

  return plaintext;
}

}  // namespace security
}  // namespace ads
