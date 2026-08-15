#include "neurotaskfm/tools.h"

#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include <nlohmann/json.hpp>

namespace neurotaskfm {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;
using tcp = asio::ip::tcp;

std::string mime_type(const std::filesystem::path& path) {
  const auto extension = path.extension().string();
  if (extension == ".html") return "text/html; charset=utf-8";
  if (extension == ".js") return "application/javascript; charset=utf-8";
  if (extension == ".css") return "text/css; charset=utf-8";
  if (extension == ".json") return "application/json";
  if (extension == ".svg") return "image/svg+xml";
  if (extension == ".png") return "image/png";
  if (extension == ".nii" || extension == ".gz") return "application/octet-stream";
  return "application/octet-stream";
}

http::response<http::string_body> response(http::status status, std::string body,
                                           std::string content_type = "application/json") {
  http::response<http::string_body> value{status, 11};
  value.set(http::field::server, "NeuroTaskFM-C++/0.1");
  value.set(http::field::content_type, content_type);
  value.set(http::field::cache_control, "no-store");
  value.body() = std::move(body);
  value.prepare_payload();
  return value;
}

http::response<http::string_body> serve(const http::request<http::string_body>& request,
                                        const std::filesystem::path& web_root) {
  const std::string target(request.target());
  if (request.method() == http::verb::get && target == "/api/health") {
    return response(http::status::ok, nlohmann::json{{"status", "ok"}, {"runtime", "libtorch-cuda"}}.dump());
  }
  if (request.method() == http::verb::get && target == "/api/functional-contrasts") {
    return response(http::status::ok, nlohmann::json{{"contrasts", nlohmann::json::array()}}.dump());
  }
  if (target.rfind("/api/", 0) == 0) {
    return response(http::status::not_implemented,
                    nlohmann::json{{"error", "This endpoint requires the native analysis job runner"}}.dump());
  }
  auto relative = target == "/" ? std::filesystem::path("index.html")
                                  : std::filesystem::path(target.substr(1));
  if (relative.string().find("..") != std::string::npos) return response(http::status::bad_request, "bad path", "text/plain");
  const auto asset_root = std::filesystem::exists(web_root / "dist/index.html")
      ? web_root / "dist" : web_root / "frontend";
  auto path = asset_root / relative;
  if (!std::filesystem::exists(path) || !std::filesystem::is_regular_file(path)) path = asset_root / "index.html";
  std::ifstream stream(path, std::ios::binary);
  if (!stream) return response(http::status::not_found, "not found", "text/plain");
  std::ostringstream body; body << stream.rdbuf();
  return response(http::status::ok, body.str(), mime_type(path));
}

void session(tcp::socket socket, const std::filesystem::path& root) {
  try {
    beast::flat_buffer buffer;
    http::request<http::string_body> request;
    http::read(socket, buffer, request);
    auto reply = serve(request, root);
    reply.keep_alive(false);
    http::write(socket, reply);
    beast::error_code error;
    socket.shutdown(tcp::socket::shutdown_send, error);
  } catch (const std::exception& error) {
    std::cerr << "web session: " << error.what() << '\n';
  }
}

}  // namespace

int run_web_server(const Arguments& arguments) {
  const auto address = asio::ip::make_address(arguments.get("address", "0.0.0.0"));
  const auto port = static_cast<unsigned short>(arguments.integer("port", 8000));
  const auto root = std::filesystem::absolute(arguments.get("root", "web_app"));
  asio::io_context context{1};
  tcp::acceptor acceptor(context, {address, port});
  std::cout << "NeuroTaskFM native web service listening on " << address << ':' << port << '\n';
  for (;;) {
    tcp::socket socket(context);
    acceptor.accept(socket);
    std::thread(session, std::move(socket), root).detach();
  }
}

}  // namespace neurotaskfm
