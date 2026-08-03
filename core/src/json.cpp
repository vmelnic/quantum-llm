#include "expert/core/json.hpp"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <limits>
#include <sstream>
#include <type_traits>

namespace expert::core::json {
namespace {

class Parser {
  public:
    explicit Parser(std::string_view input) : input_(input) {}

    Value ParseDocument() {
        SkipWhitespace();
        Value value = ParseValue();
        SkipWhitespace();
        if (position_ != input_.size()) {
            Fail("trailing content");
        }
        return value;
    }

  private:
    [[noreturn]] void Fail(std::string_view message) const {
        throw Error("JSON byte " + std::to_string(position_) + ": " +
                    std::string(message));
    }

    void SkipWhitespace() {
        while (position_ < input_.size()) {
            const char c = input_[position_];
            if (c != ' ' && c != '\n' && c != '\r' && c != '\t') {
                return;
            }
            ++position_;
        }
    }

    bool Consume(char expected) {
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void Expect(std::string_view token) {
        if (input_.substr(position_, token.size()) != token) {
            Fail("expected " + std::string(token));
        }
        position_ += token.size();
    }

    Value ParseValue() {
        if (position_ >= input_.size()) {
            Fail("expected a value");
        }
        switch (input_[position_]) {
            case 'n': Expect("null"); return Value{};
            case 't': Expect("true"); return Value(Value::Storage(true));
            case 'f': Expect("false"); return Value(Value::Storage(false));
            case '"': return Value(Value::Storage(ParseString()));
            case '[': return Value(Value::Storage(ParseArray()));
            case '{': return Value(Value::Storage(ParseObject()));
            default:
                if (input_[position_] == '-' ||
                    std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                    return ParseNumber();
                }
                Fail("unexpected token");
        }
    }

    static void AppendUtf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7fU) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ffU) {
            output.push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else if (codepoint <= 0xffffU) {
            output.push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        } else {
            output.push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
            output.push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
        }
    }

    std::uint32_t ParseHex4() {
        if (position_ + 4U > input_.size()) {
            Fail("truncated unicode escape");
        }
        std::uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            const char c = input_[position_++];
            value <<= 4U;
            if (c >= '0' && c <= '9') value |= static_cast<std::uint32_t>(c - '0');
            else if (c >= 'a' && c <= 'f') value |= static_cast<std::uint32_t>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') value |= static_cast<std::uint32_t>(c - 'A' + 10);
            else Fail("invalid unicode escape");
        }
        return value;
    }

    std::string ParseString() {
        if (!Consume('"')) Fail("expected string");
        std::string output;
        while (position_ < input_.size()) {
            const unsigned char raw = static_cast<unsigned char>(input_[position_++]);
            if (raw == '"') return output;
            if (raw < 0x20U) Fail("control character in string");
            if (raw != '\\') {
                output.push_back(static_cast<char>(raw));
                continue;
            }
            if (position_ >= input_.size()) Fail("truncated escape");
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': output.push_back('"'); break;
                case '\\': output.push_back('\\'); break;
                case '/': output.push_back('/'); break;
                case 'b': output.push_back('\b'); break;
                case 'f': output.push_back('\f'); break;
                case 'n': output.push_back('\n'); break;
                case 'r': output.push_back('\r'); break;
                case 't': output.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = ParseHex4();
                    if (codepoint >= 0xd800U && codepoint <= 0xdbffU) {
                        if (position_ + 2U > input_.size() || input_[position_] != '\\' ||
                            input_[position_ + 1U] != 'u') {
                            Fail("high surrogate without low surrogate");
                        }
                        position_ += 2U;
                        const std::uint32_t low = ParseHex4();
                        if (low < 0xdc00U || low > 0xdfffU) {
                            Fail("invalid low surrogate");
                        }
                        codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                                    (low - 0xdc00U);
                    } else if (codepoint >= 0xdc00U && codepoint <= 0xdfffU) {
                        Fail("unpaired low surrogate");
                    }
                    AppendUtf8(output, codepoint);
                    break;
                }
                default: Fail("invalid escape");
            }
        }
        Fail("unterminated string");
    }

    Value::Array ParseArray() {
        Consume('[');
        Value::Array result;
        SkipWhitespace();
        if (Consume(']')) return result;
        while (true) {
            SkipWhitespace();
            result.push_back(ParseValue());
            SkipWhitespace();
            if (Consume(']')) return result;
            if (!Consume(',')) Fail("expected ',' or ']'");
        }
    }

    Value::Object ParseObject() {
        Consume('{');
        Value::Object result;
        SkipWhitespace();
        if (Consume('}')) return result;
        while (true) {
            SkipWhitespace();
            if (position_ >= input_.size() || input_[position_] != '"') {
                Fail("expected object key");
            }
            std::string key = ParseString();
            SkipWhitespace();
            if (!Consume(':')) Fail("expected ':'");
            SkipWhitespace();
            auto [unused, inserted] = result.emplace(std::move(key), ParseValue());
            if (!inserted) Fail("duplicate object key");
            SkipWhitespace();
            if (Consume('}')) return result;
            if (!Consume(',')) Fail("expected ',' or '}'");
        }
    }

    Value ParseNumber() {
        const std::size_t begin = position_;
        const bool negative = Consume('-');
        if (position_ >= input_.size()) Fail("truncated integer");
        if (input_[position_] == '0') {
            ++position_;
            if (position_ < input_.size() &&
                std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                Fail("leading zero in integer");
            }
        } else {
            if (input_[position_] < '1' || input_[position_] > '9') {
                Fail("invalid integer");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                ++position_;
            }
        }
        bool non_integral = false;
        if (position_ < input_.size() && input_[position_] == '.') {
            non_integral = true;
            ++position_;
            if (position_ >= input_.size() ||
                !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                Fail("fraction has no digits");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        if (position_ < input_.size() &&
            (input_[position_] == 'e' || input_[position_] == 'E')) {
            non_integral = true;
            ++position_;
            if (position_ < input_.size() &&
                (input_[position_] == '+' || input_[position_] == '-')) ++position_;
            if (position_ >= input_.size() ||
                !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                Fail("exponent has no digits");
            }
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) ++position_;
        }
        const auto token = input_.substr(begin, position_ - begin);
        if (non_integral) {
            return Value(Value::Storage(Value::Number{std::string(token)}));
        }
        if (negative) {
            std::int64_t value{};
            const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
            if (error != std::errc{} || end != token.data() + token.size()) {
                Fail("signed integer out of range");
            }
            return Value(Value::Storage(value));
        }
        std::uint64_t value{};
        const auto [end, error] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (error != std::errc{} || end != token.data() + token.size()) {
            Fail("unsigned integer out of range");
        }
        return Value(Value::Storage(value));
    }

    std::string_view input_;
    std::size_t position_{};
};

void AppendEscaped(std::string& output, std::string_view text) {
    output.push_back('"');
    constexpr char kHex[] = "0123456789abcdef";
    for (const unsigned char c : text) {
        switch (c) {
            case '"': output += "\\\""; break;
            case '\\': output += "\\\\"; break;
            case '\b': output += "\\b"; break;
            case '\f': output += "\\f"; break;
            case '\n': output += "\\n"; break;
            case '\r': output += "\\r"; break;
            case '\t': output += "\\t"; break;
            default:
                if (c < 0x20U) {
                    output += "\\u00";
                    output.push_back(kHex[(c >> 4U) & 0x0fU]);
                    output.push_back(kHex[c & 0x0fU]);
                } else {
                    output.push_back(static_cast<char>(c));
                }
        }
    }
    output.push_back('"');
}

void AppendCanonical(std::string& output, const Value& value) {
    std::visit([&output](const auto& item) {
        using T = std::decay_t<decltype(item)>;
        if constexpr (std::is_same_v<T, std::nullptr_t>) output += "null";
        else if constexpr (std::is_same_v<T, bool>) output += item ? "true" : "false";
        else if constexpr (std::is_same_v<T, std::int64_t> ||
                           std::is_same_v<T, std::uint64_t>) output += std::to_string(item);
        else if constexpr (std::is_same_v<T, Value::Number>) output += item.token;
        else if constexpr (std::is_same_v<T, std::string>) AppendEscaped(output, item);
        else if constexpr (std::is_same_v<T, Value::Array>) {
            output.push_back('[');
            bool first = true;
            for (const auto& child : item) {
                if (!first) output.push_back(',');
                first = false;
                AppendCanonical(output, child);
            }
            output.push_back(']');
        } else {
            output.push_back('{');
            bool first = true;
            for (const auto& [key, child] : item) {
                if (!first) output.push_back(',');
                first = false;
                AppendEscaped(output, key);
                output.push_back(':');
                AppendCanonical(output, child);
            }
            output.push_back('}');
        }
    }, value.data);
}

}  // namespace

const Value::Object& Value::AsObject(std::string_view where) const {
    const auto* value = std::get_if<Object>(&data);
    if (value == nullptr) throw Error(std::string(where) + " must be an object");
    return *value;
}
Value::Object& Value::AsObject(std::string_view where) {
    auto* value = std::get_if<Object>(&data);
    if (value == nullptr) throw Error(std::string(where) + " must be an object");
    return *value;
}
const Value::Array& Value::AsArray(std::string_view where) const {
    const auto* value = std::get_if<Array>(&data);
    if (value == nullptr) throw Error(std::string(where) + " must be an array");
    return *value;
}
const std::string& Value::AsString(std::string_view where) const {
    const auto* value = std::get_if<std::string>(&data);
    if (value == nullptr) throw Error(std::string(where) + " must be a string");
    return *value;
}
bool Value::AsBool(std::string_view where) const {
    const auto* value = std::get_if<bool>(&data);
    if (value == nullptr) throw Error(std::string(where) + " must be boolean");
    return *value;
}
std::uint64_t Value::AsU64(std::string_view where) const {
    if (const auto* value = std::get_if<std::uint64_t>(&data)) return *value;
    if (const auto* value = std::get_if<std::int64_t>(&data); value != nullptr && *value >= 0) {
        return static_cast<std::uint64_t>(*value);
    }
    throw Error(std::string(where) + " must be a non-negative integer");
}

Value Parse(std::string_view text) { return Parser(text).ParseDocument(); }

std::string Canonicalize(const Value& value) {
    std::string output;
    AppendCanonical(output, value);
    return output;
}

std::string Canonicalize(std::string_view text) { return Canonicalize(Parse(text)); }

const Value& Required(const Value::Object& object, std::string_view key,
                      std::string_view where) {
    const auto iterator = object.find(key);
    if (iterator == object.end()) {
        throw Error(std::string(where) + " is missing required key '" +
                    std::string(key) + "'");
    }
    return iterator->second;
}

void RequireExactKeys(const Value::Object& object,
                      std::initializer_list<std::string_view> required,
                      std::initializer_list<std::string_view> optional,
                      std::string_view where) {
    for (const auto key : required) static_cast<void>(Required(object, key, where));
    for (const auto& [key, unused] : object) {
        const auto known = [key_view = std::string_view(key)](std::string_view candidate) {
            return key_view == candidate;
        };
        if (std::none_of(required.begin(), required.end(), known) &&
            std::none_of(optional.begin(), optional.end(), known)) {
            throw Error(std::string(where) + " contains unknown key '" + key + "'");
        }
    }
}

}  // namespace expert::core::json
