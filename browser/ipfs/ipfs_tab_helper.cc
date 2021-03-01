/* Copyright (c) 2020 The Brave Authors. All rights reserved.
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this file,
 * You can obtain one at http://mozilla.org/MPL/2.0/. */

#include "brave/browser/ipfs/ipfs_tab_helper.h"

#include <string>
#include <utility>
#include <vector>

#include "base/bind.h"
#include "base/callback_helpers.h"
#include "base/containers/contains.h"
#include "base/location.h"
#include "base/single_thread_task_runner.h"
#include "base/strings/string_number_conversions.h"
#include "base/threading/thread_task_runner_handle.h"
#include "brave/browser/ipfs/ipfs_host_resolver.h"
#include "brave/browser/ipfs/ipfs_service_factory.h"
#include "brave/components/ipfs/ipfs_constants.h"
#include "brave/components/ipfs/ipfs_utils.h"
#include "brave/components/ipfs/pref_names.h"
#include "chrome/browser/browser_process.h"
#include "chrome/browser/net/system_network_context_manager.h"
#include "chrome/browser/shell_integration.h"
#include "chrome/common/channel_info.h"
#include "components/prefs/pref_service.h"
#include "components/user_prefs/user_prefs.h"
#include "content/public/browser/browser_context.h"
#include "content/public/browser/browser_task_traits.h"
#include "content/public/browser/browser_thread.h"
#include "content/public/browser/navigation_handle.h"
#include "content/public/browser/storage_partition.h"
#include "content/public/browser/web_contents_delegate.h"
#include "net/http/http_status_code.h"

namespace {

// DNSLink values are of the form: dnslink=<value>
// https://dnslink.io/#dnslink-format
const char kDnsLinkHeader[] = "dnslink";

// We have to check both domain and _dnslink.domain
// https://dnslink.io/#can-i-use-dnslink-in-non-dns-systems
const char kDnsDomainPrefix[] = "_dnslink.";

// Sets current executable as default protocol handler in a system.
void SetupIPFSProtocolHandler(const std::string& protocol) {
  auto isDefaultCallback = [](const std::string& protocol,
                              shell_integration::DefaultWebClientState state) {
    if (state == shell_integration::IS_DEFAULT) {
      VLOG(1) << protocol << " already has a handler";
      return;
    }
    VLOG(1) << "Set as default handler for " << protocol;
    // The worker pointer is reference counted. While it is running, the
    // sequence it runs on will hold references it will be automatically
    // freed once all its tasks have finished.
    base::MakeRefCounted<shell_integration::DefaultProtocolClientWorker>(
        protocol)
        ->StartSetAsDefault(base::NullCallback());
  };

  base::MakeRefCounted<shell_integration::DefaultProtocolClientWorker>(protocol)
      ->StartCheckIsDefault(base::BindOnce(isDefaultCallback, protocol));
}

// Expects dns TXT record in format: name=value
std::string GetDNSRecordValue(const std::vector<std::string>& text_results,
                              const std::string& name) {
  for (const auto& txt : text_results) {
    if (!base::StartsWith(txt, name))
      continue;
    std::vector<std::string> tokens = base::SplitString(
        txt, "=", base::KEEP_WHITESPACE, base::SPLIT_WANT_ALL);
    if (!tokens.size())
      continue;
    return tokens.back();
  }
  return std::string();
}
}  // namespace

namespace ipfs {

IPFSTabHelper::~IPFSTabHelper() = default;

IPFSTabHelper::IPFSTabHelper(content::WebContents* web_contents)
    : content::WebContentsObserver(web_contents) {
  pref_service_ = user_prefs::UserPrefs::Get(web_contents->GetBrowserContext());
  auto* storage_partition = content::BrowserContext::GetDefaultStoragePartition(
      web_contents->GetBrowserContext());

  resolvers_.reserve(2);
  resolvers_.push_back(std::make_unique<IPFSHostResolver>(
      storage_partition->GetNetworkContext(), kDnsDomainPrefix));
  resolvers_.push_back(std::make_unique<IPFSHostResolver>(
      storage_partition->GetNetworkContext()));
}

// static
bool IPFSTabHelper::MaybeCreateForWebContents(
    content::WebContents* web_contents) {
  if (!ipfs::IpfsServiceFactory::GetForContext(
          web_contents->GetBrowserContext())) {
    return false;
  }

  CreateForWebContents(web_contents);
  return true;
}

void IPFSTabHelper::HostResolvedCallback(
    const std::string& host,
    const std::vector<std::string>& text_results) {
  GURL current = web_contents()->GetURL();
  if (current.host() != host || !current.SchemeIsHTTPOrHTTPS())
    return;
  std::string dnslink = GetDNSRecordValue(text_results, kDnsLinkHeader);
  if (dnslink.empty())
    return;
  ipfs_resolved_host_ = host;
  if (pref_service_->GetBoolean(kIPFSAutoRedirectDNSLink)) {
    content::OpenURLParams params(GetIPFSResolvedURL(), content::Referrer(),
                                  WindowOpenDisposition::CURRENT_TAB,
                                  ui::PAGE_TRANSITION_LINK, false);
    web_contents()->OpenURL(params);
    return;
  }
  UpdateLocationBar();
}

void IPFSTabHelper::UpdateLocationBar() {
  if (web_contents()->GetDelegate())
    web_contents()->GetDelegate()->NavigationStateChanged(
        web_contents(), content::INVALIDATE_TYPE_URL);
}

GURL IPFSTabHelper::GetIPFSResolvedURL() const {
  if (ipfs_resolved_host_.empty())
    return GURL();
  GURL current = web_contents()->GetURL();
  GURL::Replacements replacements;
  replacements.SetSchemeStr(kIPNSScheme);
  return current.ReplaceComponents(replacements);
}

bool IPFSTabHelper::HasActiveResolverForHost(const std::string& host) const {
  for (auto& resolver : resolvers_) {
    if (!resolver->IsRunning())
      continue;
    if (resolver->host() == host)
      return true;
  }
  return false;
}

void IPFSTabHelper::ResolveIPFSLink() {
  GURL current = web_contents()->GetURL();
  if (!current.SchemeIsHTTPOrHTTPS()) {
    return;
  }
  auto resolve_method = static_cast<ipfs::IPFSResolveMethodTypes>(
      pref_service_->GetInteger(kIPFSResolveMethod));
  if (resolve_method == ipfs::IPFSResolveMethodTypes::IPFS_DISABLED ||
      resolve_method == ipfs::IPFSResolveMethodTypes::IPFS_ASK)
    return;

  if (ipfs_resolved_host_ == current.host())
    return;

  if (HasActiveResolverForHost(current.host()))
    return;

  const auto& host_port_pair = net::HostPortPair::FromURL(current);
  for (auto& resolver : resolvers_) {
    if (resolver->IsRunning())
      resolver->Reset();

    auto resolved_callback = base::BindOnce(
        &IPFSTabHelper::HostResolvedCallback, weak_ptr_factory_.GetWeakPtr());
    const auto& key =
        web_contents()->GetMainFrame()
            ? web_contents()->GetMainFrame()->GetNetworkIsolationKey()
            : net::NetworkIsolationKey();
    resolver->Resolve(host_port_pair, key, net::DnsQueryType::TXT,
                      std::move(resolved_callback));
  }
}

void IPFSTabHelper::DidFinishNavigation(content::NavigationHandle* handle) {
  DCHECK(handle);
  if (!handle->IsInMainFrame() || !handle->HasCommitted() ||
      !handle->GetResponseHeaders()) {
    return;
  }

  if (!handle->IsSameDocument() && !ipfs_resolved_host_.empty()) {
    ipfs_resolved_host_.erase();
    UpdateLocationBar();
  }

  int response_code = handle->GetResponseHeaders()->response_code();
  if (ipfs_resolved_host_.empty() &&
      response_code >= net::HttpStatusCode::HTTP_INTERNAL_SERVER_ERROR &&
      response_code <= net::HttpStatusCode::HTTP_GATEWAY_TIMEOUT) {
    ResolveIPFSLink();
    return;
  }

  if (handle->GetResponseHeaders()->HasHeader("x-ipfs-path") &&
      !handle->IsSameDocument()) {
    ResolveIPFSLink();
  }

  auto resolve_method = static_cast<ipfs::IPFSResolveMethodTypes>(
      pref_service_->GetInteger(kIPFSResolveMethod));
  auto* browser_context = web_contents()->GetBrowserContext();
  if (resolve_method == ipfs::IPFSResolveMethodTypes::IPFS_ASK &&
      handle->GetResponseHeaders()->HasHeader("x-ipfs-path") &&
      IsDefaultGatewayURL(GURL(handle->GetURL()), browser_context)) {
    auto infobar_count = pref_service_->GetInteger(kIPFSInfobarCount);
    if (!infobar_count) {
      pref_service_->SetInteger(kIPFSInfobarCount, infobar_count + 1);
      SetupIPFSProtocolHandler(ipfs::kIPFSScheme);
      SetupIPFSProtocolHandler(ipfs::kIPNSScheme);
    }
  }
}

WEB_CONTENTS_USER_DATA_KEY_IMPL(IPFSTabHelper)

}  // namespace ipfs
