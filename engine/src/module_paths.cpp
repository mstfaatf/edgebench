#include "edgebench/module_paths.hpp"

#include <array>
#include <charconv>
#include <vector>

namespace edgebench {

namespace {

constexpr wchar_t kSep = L'\\';

wchar_t lower_ascii(wchar_t c) {
    return (c >= L'A' && c <= L'Z') ? static_cast<wchar_t>(c - L'A' + L'a') : c;
}

std::wstring lower_ascii(std::wstring_view s) {
    std::wstring out(s);
    for (wchar_t& c : out)
        c = lower_ascii(c);
    return out;
}

bool starts_with_ci(std::wstring_view s, std::wstring_view prefix) {
    if (s.size() < prefix.size())
        return false;
    for (size_t i = 0; i < prefix.size(); ++i) {
        if (lower_ascii(s[i]) != lower_ascii(prefix[i]))
            return false;
    }
    return true;
}

}  // namespace

std::wstring normalize_path(std::wstring_view path) {
    std::wstring p(path);
    for (wchar_t& c : p) {
        if (c == L'/')
            c = kSep;
    }

    // Extended-length prefixes: "\\?\UNC\server\share" -> "\\server\share", "\\?\C:\x" -> "C:\x".
    std::wstring_view rest = p;
    std::wstring root;
    if (starts_with_ci(rest, L"\\\\?\\UNC\\")) {
        root = L"\\\\";
        rest.remove_prefix(8);
    } else if (starts_with_ci(rest, L"\\\\?\\")) {
        rest.remove_prefix(4);
    }
    if (root.empty()) {
        if (rest.size() >= 2 && rest[0] == kSep && rest[1] == kSep) {
            root = L"\\\\";  // UNC path
            rest.remove_prefix(2);
        } else if (rest.size() >= 2 && rest[1] == L':') {
            root.assign(rest.substr(0, 2));  // drive, e.g. "C:"
            rest.remove_prefix(2);
            if (!rest.empty() && rest[0] == kSep)
                root += kSep;
        } else if (!rest.empty() && rest[0] == kSep) {
            root = L"\\";  // rooted on the current drive
        }
    }
    const bool absolute = !root.empty();

    std::vector<std::wstring_view> segments;
    while (!rest.empty()) {
        const size_t end = rest.find(kSep);
        const std::wstring_view seg = rest.substr(0, end);
        rest.remove_prefix(end == std::wstring_view::npos ? rest.size() : end + 1);
        if (seg.empty() || seg == L".")
            continue;
        if (seg == L"..") {
            if (!segments.empty() && segments.back() != L"..") {
                segments.pop_back();
            } else if (!absolute) {
                segments.push_back(seg);  // a relative path may climb above its start
            }
            continue;
        }
        segments.push_back(seg);
    }

    std::wstring out = root;
    for (size_t i = 0; i < segments.size(); ++i) {
        if (i > 0 || (!out.empty() && out.back() != kSep))
            out += kSep;
        out += segments[i];
    }
    // Drop a trailing separator left by a bare root ("C:\" -> "c:", "\" -> "").
    while (!out.empty() && out.back() == kSep && out != L"\\\\")
        out.pop_back();
    return lower_ascii(out);
}

bool is_path_within(std::wstring_view path, std::wstring_view dir) {
    const std::wstring p = normalize_path(path);
    const std::wstring d = normalize_path(dir);
    if (p.empty() || d.empty())
        return false;
    if (p == d)
        return true;
    return p.size() > d.size() && p.compare(0, d.size(), d) == 0 &&
           (p[d.size()] == kSep || d.back() == kSep);
}

bool is_same_path(std::wstring_view a, std::wstring_view b) {
    const std::wstring na = normalize_path(a);
    return !na.empty() && na == normalize_path(b);
}

bool is_ort_or_qnn_module(std::wstring_view file_name) {
    if (file_name.find_first_of(L"\\/:") != std::wstring_view::npos)
        return false;
    const std::wstring name = lower_ascii(file_name);
    const std::wstring_view n = name;
    if (!n.ends_with(L".dll"))
        return false;
    return n.starts_with(L"onnxruntime") || n.starts_with(L"qnn");
}

std::optional<Version> parse_version(std::string_view text) {
    std::array<int, 3> parts{};
    for (size_t i = 0; i < parts.size(); ++i) {
        const size_t end = (i + 1 < parts.size()) ? text.find('.') : text.size();
        if (end == std::string_view::npos)
            return std::nullopt;
        const std::string_view part = text.substr(0, end);
        if (part.empty() || part.find_first_not_of("0123456789") != std::string_view::npos)
            return std::nullopt;
        const auto [ptr, ec] = std::from_chars(part.data(), part.data() + part.size(), parts[i]);
        if (ec != std::errc() || ptr != part.data() + part.size())
            return std::nullopt;
        text.remove_prefix(end == text.size() ? end : end + 1);
    }
    return Version{parts[0], parts[1], parts[2]};
}

bool version_matches(std::string_view found, std::string_view expected) {
    const std::optional<Version> f = parse_version(found);
    const std::optional<Version> e = parse_version(expected);
    return f && e && *f == *e;
}

}  // namespace edgebench
