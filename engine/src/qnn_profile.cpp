#include "edgebench/qnn_profile.hpp"

#include <charconv>

namespace edgebench {

namespace {

Error unreadable(const std::string& what) {
    return Error{ExitCode::internal_error, "qnn_profile_unreadable",
                 "cannot read the QNN profiling output: " + what};
}

std::string_view strip_cr(std::string_view line) {
    if (!line.empty() && line.back() == '\r')
        line.remove_suffix(1);
    return line;
}

}  // namespace

Result<std::vector<QnnProfileEvent>> parse_qnn_profile_csv(std::string_view text) {
    std::vector<QnnProfileEvent> events;
    std::size_t start = 0;
    int line_number = 0;
    while (start < text.size()) {
        std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos)
            end = text.size();
        const std::string_view line = strip_cr(text.substr(start, end - start));
        start = end + 1;
        ++line_number;
        if (line_number == 1) {
            if (line != kQnnProfileHeader)
                return unreadable("unexpected header '" + std::string(line) + "'");
            continue;
        }
        if (line.empty())
            continue;
        std::vector<std::string_view> fields;
        std::size_t field_start = 0;
        while (fields.size() < 6) {
            const std::size_t comma = line.find(',', field_start);
            if (comma == std::string_view::npos)
                break;
            fields.push_back(line.substr(field_start, comma - field_start));
            field_start = comma + 1;
        }
        if (fields.size() < 6)
            return unreadable("line " + std::to_string(line_number) + " has fewer than 7 fields");
        fields.push_back(line.substr(field_start));  // the identifier keeps its commas

        QnnProfileEvent e;
        const std::string_view time = fields[2];
        const auto [ptr, ec] = std::from_chars(time.data(), time.data() + time.size(), e.time);
        if (ec != std::errc() || ptr != time.data() + time.size())
            return unreadable("line " + std::to_string(line_number) + " has the time '" +
                              std::string(time) + "'");
        e.message = std::string(fields[1]);
        e.unit = std::string(fields[3]);
        e.level = std::string(fields[5]);
        e.identifier = std::string(fields[6]);
        events.push_back(std::move(e));
    }
    if (line_number == 0)
        return unreadable("the file is empty");
    return events;
}

QnnProfileSummary summarize_qnn_profile(const std::vector<QnnProfileEvent>& events, int skip) {
    QnnProfileSummary s;
    std::map<std::string, double> current;  // execute events of the inference being read
    for (const QnnProfileEvent& e : events) {
        // "(execute) time", "(execute excluding wait) time": per inference.
        if (e.message == "EXECUTE" || e.identifier.find("(execute") != std::string::npos) {
            if (e.unit == "US" && e.level == "ROOT")
                current[e.identifier] = e.time / 1000.0;
            if (e.message == "EXECUTE") {
                if (s.inferences >= skip) {
                    for (const auto& [id, ms] : current)
                        s.execute_ms[id].push_back(ms);
                }
                ++s.inferences;
                current.clear();
            }
        } else if (e.unit == "US" && e.level == "ROOT") {
            s.init_ms.emplace_back(e.identifier, e.time / 1000.0);
        }
    }
    return s;
}

}  // namespace edgebench
