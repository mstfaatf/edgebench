#include "edgebench/qnn_backend_info.hpp"

#include <charconv>

namespace edgebench {

namespace {

std::vector<std::string_view> split(std::string_view text, char separator) {
    std::vector<std::string_view> parts;
    std::size_t start = 0;
    while (true) {
        const std::size_t end = text.find(separator, start);
        parts.push_back(text.substr(start, end == std::string_view::npos ? end : end - start));
        if (end == std::string_view::npos)
            return parts;
        start = end + 1;
    }
}

std::optional<std::uint32_t> parse_uint(std::string_view text) {
    if (text.empty())
        return std::nullopt;
    std::uint32_t value = 0;
    const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (ec != std::errc() || end != text.data() + text.size())
        return std::nullopt;
    return value;
}

// "2.52.0": exactly three numeric parts.
bool is_version(std::string_view text) {
    const std::vector<std::string_view> parts = split(text, '.');
    if (parts.size() != 3)
        return false;
    for (std::string_view p : parts)
        if (!parse_uint(p))
            return false;
    return true;
}

std::optional<std::vector<std::uint32_t>> parse_list(std::string_view text) {
    std::vector<std::uint32_t> values;
    for (std::string_view p : split(text, ',')) {
        const std::optional<std::uint32_t> v = parse_uint(p);
        if (!v)
            return std::nullopt;
        values.push_back(*v);
    }
    return values;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    while (!s.empty() &&
           (s.back() == ' ' || s.back() == '\t' || s.back() == '\r' || s.back() == '\n'))
        s.remove_suffix(1);
    return s;
}

}  // namespace

std::optional<std::string> find_backend_build_id(const std::vector<std::string>& messages) {
    constexpr std::string_view kPrefix = "Backend build version:";
    std::optional<std::string> found;
    for (const std::string& message : messages) {
        const std::size_t at = message.find(kPrefix);
        if (at == std::string::npos)
            continue;
        std::string_view rest = trim(std::string_view(message).substr(at + kPrefix.size()));
        rest = rest.substr(0, rest.find_first_of(" \t\r\n"));
        if (!rest.empty())
            found = std::string(rest);
    }
    return found;
}

std::optional<std::string> sdk_version_from_build_id(std::string_view build_id) {
    if (build_id.size() < 2 || build_id.front() != 'v')
        return std::nullopt;
    const std::vector<std::string_view> parts = split(build_id.substr(1), '.');
    if (parts.size() != 4)
        return std::nullopt;
    for (std::string_view p : parts) {
        // The build part (e.g. 261104115025) does not fit 32 bits: digits only.
        if (p.empty() || p.find_first_not_of("0123456789") != std::string_view::npos)
            return std::nullopt;
    }
    return std::string(parts[0]) + "." + std::string(parts[1]) + "." + std::string(parts[2]);
}

std::optional<QnnCompatibilityInfo> parse_qnn_compatibility_info(std::string_view info) {
    const std::vector<std::string_view> f = split(info, ':');
    if (f.size() != 8 || f[0] != "v2")
        return std::nullopt;
    QnnCompatibilityInfo out;
    const std::optional<std::uint32_t> backend = parse_uint(f[1]);
    if (!backend || !is_version(f[2]) || !is_version(f[3]))
        return std::nullopt;
    out.backend_id = *backend;
    out.sdk_version = std::string(f[2]);
    out.backend_api_version = std::string(f[3]);
    std::optional<std::vector<std::uint32_t>> archs = parse_list(f[4]);
    std::optional<std::vector<std::uint32_t>> socs = parse_list(f[5]);
    std::optional<std::vector<std::uint32_t>> vtcm = parse_list(f[6]);
    if (!archs || !socs || !vtcm || socs->size() != archs->size() ||
        vtcm->size() != archs->size() || (f[7] != "0" && f[7] != "1"))
        return std::nullopt;
    out.htp_archs = std::move(*archs);
    out.soc_models = std::move(*socs);
    out.vtcm_mbs = std::move(*vtcm);
    out.htp_user_driver = f[7] == "1";
    return out;
}

std::optional<std::string> htp_arch_name(const QnnCompatibilityInfo& info) {
    if (info.htp_archs.size() != 1 || info.htp_archs.front() == 0)
        return std::nullopt;
    return "v" + std::to_string(info.htp_archs.front());
}

std::optional<std::string> compatibility_verdict_name(int value) {
    switch (value) {
        case 0:
            return "not_applicable";
        case 1:
            return "optimal";
        case 2:
            return "prefer_recompilation";
        case 3:
            return "unsupported";
        default:
            return std::nullopt;
    }
}

std::optional<std::string> recompilation_reason(const QnnCompatibilityInfo& info) {
    for (std::uint32_t mb : info.vtcm_mbs) {
        if (mb == 0) {
            return "The QNN EP rates an EP context optimal only if the VTCM size stored in its "
                   "compatibility info equals the device's; this context stores 0 (the vtcm_mb "
                   "option was not set, so the EP used its default), which never matches.";
        }
    }
    return std::nullopt;
}

}  // namespace edgebench
