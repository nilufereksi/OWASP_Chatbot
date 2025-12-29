# OWASP Güvenlik Politikası ve Prosedürleri

## 1. Giriş ve Kapsam
Bu doküman, XYZ Şirketi'nin yazılım geliştirme yaşam döngüsü boyunca izlemesi gereken asgari güvenlik standartlarını tanımlar. Tüm yazılımcılar ve QA ekipleri bu prosedürlere uymak zorundadır.

## 2. OWASP Top 10 (2021) Kontrolleri

### A01:2021 Bozuk Erişim Kontrolü (Broken Access Control)
* **Tanım:** Kullanıcıların yetkili olmadıkları fonksiyonlara veya verilere erişimi.
* **Önleme Prosedürü:** Her istekte sunucu tarafında yetkilendirme kontrolü (RBAC veya ABAC) uygulanmalıdır. **IDOR (Insecure Direct Object Reference)** hataları için, girdi yerine kullanıcının yetkilendirildiği nesne ID'leri kullanılmalıdır.

### A02:2021 Kriptografik Hatalar (Cryptographic Failures)
* **Tanım:** Hassas verilerin (parolalar, kimlik bilgileri, finansal veriler) uygunsuz şekilde korunması veya şifreleme eksikliği.
* **Önleme Prosedürü:** Atıl Durumdaki Veri (Data at Rest): Veritabanlarında ve dosya sistemlerinde depolanan hassas veriler için AES-256 gibi modern ve güçlü şifreleme algoritmaları kullanılmalıdır. İletimdeki Veri (Data in Transit): Tüm iletişimde (API çağrıları, web trafiği) yalnızca TLS 1.2 veya üzeri zorunlu tutulmalı, eski ve zayıf şifre paketleri (cipher suites) kesinlikle devre dışı bırakılmalıdır.

### A03:2021 Enjeksiyon (Injection)
* **Tanım:** Güvenilmeyen verinin komut yorumlayıcıya gönderilmesi.
* **Önleme Prosedürü:** SQL enjeksiyonu için **parametreli sorgular (Prepared Statements)** kullanılmalıdır. Komut satırı enjeksiyonu için, kullanıcı girdileri hiçbir zaman doğrudan komut satırı argümanı olarak geçirilmemelidir.

### A04:2021 Güvenli Olmayan Tasarım (Insecure Design)
* **Tanım:** Yazılımın doğası gereği güvenlik risklerine yol açan eksik veya hatalı kontrol tasarımı. Bu, kod seviyesi değil, mimari seviyesindeki zafiyetlerdir.
* **Önleme Prosedürü:** Varsayılan Reddetme (Deny by Default): Güvenlik kararları alınırken, açıkça izin verilmeyen her şeyin otomatik olarak reddedilmesi ilkesi benimsenmelidir. Güven Sınırları (Trust Boundaries): Farklı güven seviyesindeki bileşenler (örneğin kullanıcı arayüzü ve backend) arasında net sınırlar belirlenmeli ve bu sınırları aşan her türlü veri, sanki güvensiz bir kaynaktan geliyormuş gibi işlenmelidir. Hız Sınırlaması (Rate Limiting): Tüm kritik API uç noktalarında ve oturum açma sayfalarında otomatik hız sınırlaması zorunludur.

### A05:2021 Güvenlik Yanlış Yapılandırması (Security Misconfiguration)
* **Tanım:** Varsayılan parolalar, güncel olmayan yazılımlar, gereksiz servisler.
* **Önleme Prosedürü:** Tüm varsayılan ayarlar, kurulumdan hemen sonra değiştirilmelidir. Kullanılmayan servisler ve özellikler devre dışı bırakılmalıdır. Hata mesajları, son kullanıcıya detaylı sistem bilgisi vermemelidir.

### A06:2021 Eskimiş ve Rentansiyonu Olmayan Bileşenler (Vulnerable and Outdated Components)
* **Tanım:** Projelerde kullanılan kütüphane, framework veya işletim sistemi bileşenlerinin güvenlik yaması yapılmamış, desteklenmeyen veya güncelliğini yitirmiş olması.
* **Önleme Prosedürü:** Bağımlılık Taraması (Dependency Scanning): Tüm projelerde, bilinen güvenlik zafiyetlerini (CVE'ler) kontrol eden otomatik tarama araçları (SCA - Software Composition Analysis) kullanılmalıdır. Yama Yönetimi: Orta ve üzeri riskli bulgular içeren tüm bileşenler, 7 iş günü içinde yamalanmalı veya güncel ve güvenli bir alternatif ile değiştirilmelidir. Kullanım Yasağı: Geliştiricilerin, ömrü dolmuş (End-of-Life) kütüphaneleri kullanması kesinlikle yasaktır.

### A07:2021 Kimlik Doğrulama Hataları (Identification and Authentication Failures)
* **Tanım:** Kullanıcı kimlik doğrulama veya oturum yönetimi mekanizmalarının hatalı uygulanması, bu durumun saldırganların kimlikleri ele geçirmesine veya oturumları taklit etmesine olanak sağlaması.
* **Önleme Prosedürü:** Çok Faktörlü Kimlik Doğrulama (MFA): Yüksek hassasiyetli tüm sistemlere erişimde MFA zorunludur. Oturum Yönetimi: Oturum kimlikleri (session IDs), güvenli ve tahmin edilemez olmalı; HTTP üzerinden asla iletilmemelidir. Boşta kalma (idle timeout) süresi 15 dakikayı geçmemelidir. Parola Politikası: Parola sıfırlama mekanizmaları, kullanıcının e-posta adresine veya doğrulanmış bir telefona gönderilen güvenli, tek kullanımlık bir token gerektirmelidir.

### A08:2021 Yazılım ve Veri Bütünlüğü Hataları (Software and Data Integrity Failures)
* **Tanım:** Yazılım mantığı, veri ve kritik meta verilerin bütünlüğüne güvenilmemesi veya bu bütünlüğün doğrulanmaması. Özellikle harici kaynaklardan gelen kod veya verilerin güvenli olmayan şekilde işlenmesi.
* **Önleme Prosedürü:** Veri Bütünlüğü Kontrolü: Yazılım güncellemeleri, kritik konfigürasyon dosyaları veya harici kaynaklardan indirilen yürütülebilir dosyalar için hash veya dijital imza ile bütünlük kontrolü zorunludur. CI/CD Güvenliği: Derleme ve dağıtım ortamları (CI/CD pipeline) sıkı erişim kontrolleriyle korunmalı ve bu ortamlarda manuel müdahaleler yasaklanmalıdır.

### A09:2021 Güvenlik Loglama ve İzleme Hataları (Security Logging and Monitoring Failures)
* **Tanım:** Güvenlik olaylarının yetersiz loglanması veya hiç loglanmaması, şüpheli faaliyetlerin zamanında tespit edilememesi.
* **Önleme Prosedürü:** Loglama Kapsamı: Tüm kimlik doğrulama denemeleri (başarılı/başarısız), yetkilendirme hataları (403), kritik veri güncellemeleri ve uygulama hataları mutlaka loglanmalıdır. Log Güvenliği: Loglar, değiştirilemezlik (immutability) ilkesi gereği, minimum 90 gün boyunca merkezi bir Log Yönetim Sistemi'nde güvenli bir şekilde saklanmalıdır. Anlık Uyarılar: Başarısız oturum açma denemeleri (örneğin 1 dakikada 5 deneme) gibi şüpheli aktiviteler için otomatik ve anlık uyarı mekanizmaları kurulmalıdır.

### A10:2021 Sunucu Tarafı İstek Sahteciliği (Server-Side Request Forgery - SSRF)
* **Tanım:** Sunucunun, kullanıcı tarafından kontrol edilen bir URL aracılığıyla harici veya dahili bir kaynağa istek yapmaya zorlanması.
* **Önleme Prosedürü:**
-Kullanıcı girdisinin tam ve katı bir şekilde validasyonu yapılmalıdır.
-Sunucunun bağlanabileceği hedefler (host, protokol, port) beyaz liste (whitelisting) yöntemiyle kısıtlanmalıdır.
-Dahili ağ adreslerine (örneğin 127.0.0.1, 192.168.x.x) istek yapılmasını engellemek için katı ağ segmentasyonu uygulanmalıdır.

### 2.1. Ek Prosedürler: Güvenli SDLC Adımları

* **Kod İncelemesi (Code Review):** Kritik ve yüksek riskli değişiklikler için, deploy edilmeden önce en az iki kıdemli geliştirici tarafından güvenlik odaklı kod incelemesi zorunludur.
* **SAST/DAST Taraması:** Tüm yeni kodlar ve güncellemeler için otomatik **SAST (Statik Uygulama Güvenlik Testi)** taraması CI/CD hattında zorunludur. Haftalık periyotlarda ise canlı sistemde **DAST (Dinamik Uygulama Güvenlik Testi)** taraması yapılmalıdır.
* **Güvenlik Kapısı (Security Gate):** SAST/DAST taramalarında **'Kritik'** veya **'Yüksek'** riskli bulgular tespit edilirse, sürüm otomatik olarak durdurulur ve bu bulgular çözülmeden üretime geçiş **kesinlikle yasaktır**.

### 2.2. Parola ve Veri Yönetimi Standartları

* **Parola Şifreleme:** Kullanıcı parolaları ve hassas API anahtarları, yalnızca **Argon2** veya **bcrypt** gibi modern ve yavaş hash algoritmalarıyla ve **Benzersiz Tuzlama (Unique Salt)** kullanılarak depolanmalıdır.
* **Veri Saklama Süresi (Retention):** Yasal gereklilikler dışında, kritik sistem logları (A09:2021) en az 90 gün, diğer hassas veriler ise sadece iş gereksinimi olduğu sürece saklanabilir.

## 3. Olay Müdahale Prosedürü (Örnek Prosedür)
Bir güvenlik açığı tespit edildiğinde aşağıdaki adımlar izlenmelidir:
1.  **Tespit ve Raporlama:** Açığı ilk bulan kişi, derhal Güvenlik Ekibi'ne bildirmelidir.
2.  **Kapsam Belirleme:** Ekip, açığın etkilediği sistemleri ve veri kapsamını belirler.
3.  **Hızlı Onarım:** Açığı kapatacak acil bir yama (hotfix) uygulanır.
4.  **Kök Neden Analizi:** Sorunun kaynağı belirlenir ve uzun vadeli önlemler alınır.

## 4. Bilgi Kapsamı ve Red Prosedürü (Negatif Testler İçin)

Bu chatbot, yalnızca bu dokümanda tanımlanan **Güvenlik Politikaları ve Prosedürleri** hakkında bilgi sağlamakla yetkilidir.

Dokümanda bulunmayan konulara (örneğin fiyatlandırma, İK politikaları, güncel hisse senedi bilgileri vb.) gelen sorulara cevap verirken, model aşağıdaki formatı kullanmalıdır:

> **[RED]:** Bu bilgi, XYZ Şirketi'nin Güvenlik Politikası dokümanının kapsamında değildir. Lütfen ilgili departmana (İK, Finans vb.) danışın.